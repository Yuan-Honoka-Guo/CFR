import argparse
import os
import torch
from torchvision import transforms
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big, GeGLU
from models.attention_retrieval import AttentionRetriever, retrieve_topk_cosine

from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score

from test_memory_bank import MemoryBankWrapper


def _load_kv_bank(bank_path, device):
    bank = torch.load(bank_path, map_location=device)
    if not isinstance(bank, dict) or "keys" not in bank or "values" not in bank:
        raise ValueError("KV memory bank not found. Build it with --bank_type kv.")
    keys = bank["keys"].to(device)
    values = bank["values"].to(device)
    return keys, values


def _sample_kv_bank(keys, values, sample_size):
    if sample_size is None or sample_size <= 0 or sample_size >= keys.shape[0]:
        return keys, values
    idx = torch.randperm(keys.shape[0], device=keys.device)[:sample_size]
    return keys[idx], values[idx]


def set_seeds(sid=42):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def infer_CFR(args):
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_attention_retrieval:
        if not args.attn_checkpoint or not args.attn_bank_path:
            raise ValueError("Attention retrieval requires --attn_checkpoint and --attn_bank_path.")
        if not args.bank_path:
            raise ValueError("Attention replacement still needs --bank_path for distance thresholding.")
    shot_tag = f'_{args.few_shot}shot' if args.few_shot is not None else ''
    run_tag = f'{args.class_name}{shot_tag}'

    # shot_tag = f'_{args.few_shot}shot' if args.few_shot is not None else ''
    # run_tag = f'{args.class_name}{shot_tag}'
    model_name = f'{run_tag}_{args.epochs_no}ep_{args.batch_size}bs'

    # Dataloaders.
    test_loader = get_data_loader("test", class_name = args.class_name, img_size = 224, dataset_path = args.dataset_path)

    # Feature extractors.
    feature_extractor = MultimodalFeatures()

    # Model instantiation. 
    CFR_2Dto3D = GeGLU(in_features=768, out_features=1152, hidden_features=(768 + 1152)//2)
    CFR_3Dto2D = GeGLU(in_features=1152, out_features=768, hidden_features=(1152 + 768)//2)

    cfr_dir = os.path.join(args.checkpoint_folder, args.class_name)
    CFR_2Dto3D_path = os.path.join(cfr_dir, f'CFR_2Dto3D_{model_name}.pth')
    CFR_3Dto2D_path = os.path.join(cfr_dir, f'CFR_3Dto2D_{model_name}.pth')

    CFR_2Dto3D.load_state_dict(torch.load(CFR_2Dto3D_path))
    CFR_3Dto2D.load_state_dict(torch.load(CFR_3Dto2D_path))

    CFR_2Dto3D.to(device), CFR_3Dto2D.to(device)

    # Make CFR modules non-trainable.
    CFR_2Dto3D.eval(), CFR_3Dto2D.eval()

    memory_bank = None
    if args.bank_path:
        memory_bank = MemoryBankWrapper(args.bank_path, device=device)

    attn_model = None
    kv_keys, kv_values = None, None
    if args.use_attention_retrieval:
        kv_keys, kv_values = _load_kv_bank(args.attn_bank_path, device=device)
        attn_model = AttentionRetriever(
            q_dim=kv_keys.shape[-1],
            k_dim=kv_keys.shape[-1],
            v_dim=kv_values.shape[-1],
            d_model=args.attn_d_model,
            dropout=0.0,
            normalize_qk=args.attn_normalize_qk,
        ).to(device)
        attn_model.load_state_dict(torch.load(args.attn_checkpoint, map_location=device))
        attn_model.eval()

    # Use box filters to approximate gaussian blur (https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf).
    w_s, w_l, w_u, w_m = 3, 5, 7, 9
    pad_s, pad_l, pad_u, pad_m = 1, 2, 3, 4
    weight_l = torch.ones(1, 1, w_l, w_l, device = device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device = device)/(w_u**2)
    weight_s = torch.ones(1, 1, w_s, w_s, device = device)/(w_s**2)
    weight_m = torch.ones(1, 1, w_m, w_m, device = device)/(w_m**2)

    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []
    # ------------ [Testing Loop] ------------ #

    # --- [Debug Init] Initialize statistics containers ---
    debug_stats = {
        "min_dists": [],       # Temporarily stores the minimum distances for each batch.
        "max_dists": [],       # Temporarily stores the maximum distances for each batch.
        "total_features": 0,   # Total number of features, equivalent to total pixels.
        "replaced_count": 0    # Number of replaced features with distance > theta.
    }
    
    theta = args.acceptance_threshold  # Distance threshold; features above it are replaced.
    # * Return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path
    for (rgb, pc, depth), gt, label, rgb_path in tqdm(test_loader, desc = f'Extracting feature from class: {args.class_name}.'):

        rgb, pc, depth = rgb.to(device), pc.to(device), depth.to(device)

        with torch.no_grad():
            # 1. Feature extraction.
            rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
            
            # 2. Crossmodal mapping (CFR).
            rgb_feat_pred = CFR_3Dto2D(xyz_patch)
            xyz_feat_pred = CFR_2Dto3D(rgb_patch)

            if memory_bank is not None:
                # 3. Flatten features for batched processing. Assumes batch_size=1 but also supports batch_size>1.
                # Flatten shape: (Total_Pixels, D)
                flat_pred = rgb_feat_pred.reshape(-1, rgb_feat_pred.shape[-1])
                flat_real = rgb_patch.reshape(-1, rgb_patch.shape[-1])
                flat_xyz = xyz_patch.reshape(-1, xyz_patch.shape[-1])
                xyz_mask_flat = (flat_xyz.sum(axis=-1) == 0)
                
                # 4. [Core] Compute distances in batch with vectorization for speed.
                # dists_pred: (Total_Pixels, 3)
                dists_pred, _ = memory_bank.get_nearest_neighbors_consine(flat_pred, k=1)
                
                # Take nearest-neighbor distances from the first column.
                min_dists = dists_pred[:, 0] # Shape: (Total_Pixels,)

                # --- [Debug Logic] Statistics collection ---
                # num_pixels = min_dists.shape[0]
                
                # A. Count distances greater than theta.
                mask_replace = min_dists >= theta
                num_replace_batch = mask_replace.sum().item()
                
                # debug_stats["total_features"] += num_pixels
                # debug_stats["replaced_count"] += num_replace_batch

                # # B. Collect the top-50 extremes for this batch as CPU lists to save GPU memory.
                # k_log = min(50, num_pixels) # Avoid errors when fewer than 50 pixels are available.
                
                # # Collect the 50 smallest values to inspect the best-matched features.
                # batch_min_topk = torch.topk(min_dists, k_log, largest=False).values.cpu().tolist()
                # debug_stats["min_dists"].extend(batch_min_topk)
                
                # # Collect the 50 largest values to inspect the most anomalous or worst-matched features.
                # batch_max_topk = torch.topk(min_dists, k_log, largest=True).values.cpu().tolist()
                # debug_stats["max_dists"].extend(batch_max_topk)
                # # --------------------------------

                # 5. Feature replacement logic, vectorized.
                if num_replace_batch > 0:
                    valid_replace = mask_replace & (~xyz_mask_flat)
                    if valid_replace.sum() > 0:
                        if args.use_attention_retrieval:
                            # Use 3D query to retrieve top-k keys/values, then attention-weight them.
                            keys, values = _sample_kv_bank(kv_keys, kv_values, args.attn_bank_sample_size)
                            query = flat_xyz[valid_replace]
                            _, topk_keys, topk_values = retrieve_topk_cosine(
                                query, keys, values, k=args.attn_retrieval_topk, chunk_size=args.attn_chunk_size
                            )
                            feats_replacement = attn_model.forward_topk(query, topk_keys, topk_values)
                            flat_pred[valid_replace] = feats_replacement
                        else:
                            # Select real features that need replacement.
                            feats_to_query = flat_real[valid_replace] # (M, D)
                            
                            # Retrieve nearest neighbors of the real features from the memory bank.
                            _, feats_replacement = memory_bank.get_nearest_neighbors_consine(feats_to_query, k=1)
                            
                            # Replace unreliable parts of the predicted features.
                            flat_pred[valid_replace] = feats_replacement.squeeze(1)
                    
                    # Reshape the updated flat predictions back into rgb_feat_pred.
                    rgb_feat_pred = flat_pred.reshape(rgb_feat_pred.shape)


            xyz_mask = (xyz_patch.sum(axis = -1) == 0) # Mask only the feature vectors that are 0 everywhere.

            cos_3d = (torch.nn.functional.normalize(xyz_feat_pred, dim = 1) - torch.nn.functional.normalize(xyz_patch, dim = 1)).pow(2).sum(1).sqrt()        
            cos_3d[xyz_mask] = 0.
            cos_3d = cos_3d.reshape(224,224)
            
            cos_2d = (torch.nn.functional.normalize(rgb_feat_pred, dim = 1) - torch.nn.functional.normalize(rgb_patch, dim = 1)).pow(2).sum(1).sqrt()        
            cos_2d[xyz_mask] = 0.
            cos_2d = cos_2d.reshape(224,224)
            
            cos_comb = (cos_2d * cos_3d) 
            cos_comb.reshape(-1)[xyz_mask] = 0.
            
            # Repeated box filters to approximate a Gaussian blur.
            cos_comb = cos_comb.reshape(1, 1, 224, 224)

            # current best: only u 1
            for _ in range(0):
                cos_comb = torch.nn.functional.conv2d(input = cos_comb, padding = pad_s, weight = weight_s) 
                
            for _ in range(1):
                cos_comb = torch.nn.functional.conv2d(input = cos_comb, padding = pad_l, weight = weight_l) 

            for _ in range(0):
                cos_comb = torch.nn.functional.conv2d(input = cos_comb, padding = pad_u, weight = weight_u) 

            for _ in range(0):
                cos_comb = torch.nn.functional.conv2d(input = cos_comb, padding = pad_m, weight = weight_m) 

            
            
            cos_comb = cos_comb.reshape(224,224)
            
            # Prediction and ground-truth accumulation.
            gts.append(gt.squeeze().cpu().detach().numpy()) # * (224,224)
            predictions.append((cos_comb / (cos_comb[cos_comb!=0].mean())).cpu().detach().numpy()) # * (224,224)
            
            # GTs.
            image_labels.append(label) # * (1,)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy()) # * (50176,)

            # Predictions.
            image_preds.append((cos_comb / torch.sqrt(cos_comb[cos_comb!=0].mean())).cpu().detach().numpy().max()) # * number
            pixel_preds.extend((cos_comb / torch.sqrt(cos_comb.mean())).flatten().cpu().detach().numpy()) # * (224,224)

            if args.produce_qualitatives:

                defect_class_str = rgb_path[0].split('/')[-3]
                image_name_str = rgb_path[0].split('/')[-1]

                save_path = f'{args.qualitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                fig, axs = plt.subplots(2,3, figsize = (7,7))

                denormalize = transforms.Compose([
                    transforms.Normalize(mean = [0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
                    transforms.Normalize(mean = [-0.485, -0.456, -0.406], std = [1., 1., 1.]),
                    ])

                rgb = denormalize(rgb)

                os.path.join(save_path, image_name_str)

                axs[0, 0].imshow(rgb.squeeze().permute(1,2,0).cpu().detach().numpy())
                axs[0, 0].set_title('RGB')

                axs[0, 1].imshow(gt.squeeze().cpu().detach().numpy())
                axs[0, 1].set_title('Ground-truth')

                axs[0, 2].imshow(depth.squeeze().permute(1,2,0).mean(axis=-1).cpu().detach().numpy())
                axs[0, 2].set_title('Depth')

                axs[1, 0].imshow(cos_3d.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 0].set_title('3D Cosine Similarity')

                axs[1, 1].imshow(cos_2d.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 1].set_title('2D Cosine Similarity')

                axs[1, 2].imshow(cos_comb.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 2].set_title('Combined Cosine Similarity')

                # Remove ticks and labels from all subplots
                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                # Adjust the layout and spacing
                plt.tight_layout()

                plt.savefig(os.path.join(save_path, image_name_str), dpi = 256)

                if args.visualize_plot:
                    plt.show()


    # if not args.use_attention_retrieval:
    #     # --- [Debug Report] Print final statistics ---
    #     print("\n" + "="*40)
    #     print(f"DEBUG REPORT: Hyperparameter Analysis (Theta={theta})")
    #     print("="*40)

    #     # 1. Calculate replacement ratio.
    #     total = debug_stats["total_features"]
    #     replaced = debug_stats["replaced_count"]
    #     ratio = (replaced / total) * 100 if total > 0 else 0

    #     print(f"Total Features Processed: {total}")
    #     print(f"Features Replaced:      {replaced}")
    #     print(f"Replacement Ratio:      {ratio:.4f}%")

    #     # 2. Distance extreme-value statistics.
    #     # Sort extremes collected from all batches again and keep the global top 50.
    #     all_mins = sorted(debug_stats["min_dists"])[:50]
    #     all_maxs = sorted(debug_stats["max_dists"], reverse=True)[:50]

    #     print("-" * 40)
    #     print("Top 10 Smallest Distances (Best Matched):")
    #     print([f"{d:.4f}" for d in all_mins[:10]]) # Print only the first 10 values as a preview.
    #     print(f"Min (0th): {all_mins[0]:.6f} | 50th Min: {all_mins[-1]:.6f}")

    #     print("-" * 40)
    #     print("Top 10 Largest Distances (Most Anomalous / Worst Matched):")
    #     print([f"{d:.4f}" for d in all_maxs[:10]]) # Print only the first 10 values as a preview.
    #     print(f"Max (0th): {all_maxs[0]:.6f} | 50th Max: {all_maxs[-1]:.6f}")
    #     print("="*40 + "\n")
    # Calculate AD&S metrics.
    au_pros, _ = calculate_au_pro(gts, predictions)
    pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    result_file_name = f'{args.quantitative_folder}/{run_tag}_{args.epochs_no}ep_{args.batch_size}bs.md'
    
    title_string = f'Metrics for class {args.class_name}{shot_tag} with {args.epochs_no}ep_{args.batch_size}bs'
    header_string = 'AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC'
    results_string = f'{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & {au_pros[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}'

    if not os.path.exists(args.quantitative_folder):
        os.makedirs(args.quantitative_folder)

    with open(result_file_name, "w") as markdown_file:
        markdown_file.write(title_string + '\n' + header_string + '\n' + results_string)

    # Print AD&S metrics.
    print(title_string)
    print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC | I-AUROC")
    print(f'  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   {au_pros[3]:.3f}  |   {pixel_rocauc:.3f} |   {image_rocauc:.3f}', end = '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Make inference with Crossmodal Feature Replacer (CFR) on a dataset.')

    parser.add_argument('--dataset_path', default = '/path/to/your/dataset/mvtec3d', type = str, 
                        help = 'Dataset path.')
    
    parser.add_argument('--class_name', default = None, type = str, choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
                                                                               'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help = 'Category name.')
    
    parser.add_argument('--checkpoint_folder', default = '/path/to/your/checkpoints/checkpoints_CFR_mvtec', type = str,
                        help = 'Path to the folder containing CFR checkpoints.')

    parser.add_argument('--qualitative_folder', default = '/path/to/your/results/qualitatives_mvtec', type = str,
                        help = 'Path to the folder in which to save the qualitatives.')

    parser.add_argument('--quantitative_folder', default = '/path/to/your/results/quantitatives_mvtec', type = str,
                        help = 'Path to the folder in which to save the quantitatives.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs used to train CFR modules.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--few_shot', default = None, type = int,
                        help = 'Number of training samples used for few-shot checkpoints.')
    
    parser.add_argument('--visualize_plot', default = False, action = 'store_true',
                        help = 'Whether to show plot or not.')
    
    parser.add_argument('--produce_qualitatives', default = False, action = 'store_true',
                        help = 'Whether to produce qualitatives or not.')
    
    parser.add_argument('--bank_path', type=str, 
                        default='/path/to/your/checkpoints/memory_bank/bagel/memory_bank_full.pt',
                        help='Path to the .pt file containing the feature tensor')
    
    parser.add_argument('--acceptance_threshold', type=float, default = 0.2,
                        help='Minimum feature distance in the memory bank greater than it will be resampled')

    parser.add_argument('--use_attention_retrieval', default=True, action='store_true',
                        help='Use attention-based retrieval to replace 2D features.')
    parser.add_argument('--attn_checkpoint', type=str, default=None,
                        help='Path to attention retriever checkpoint.')
    parser.add_argument('--attn_bank_path', type=str, default=None,
                        help='Path to KV memory bank (.pt) built with --bank_type kv.')
    parser.add_argument('--attn_d_model', default=128, type=int)
    parser.add_argument('--attn_normalize_qk', default=False, action='store_true')
    parser.add_argument('--attn_chunk_size', default=4096, type=int)
    parser.add_argument('--attn_bank_sample_size', default=20000, type=int)
    parser.add_argument('--attn_retrieval_topk', default=3, type=int)
    
    args = parser.parse_args()

    infer_CFR(args)
