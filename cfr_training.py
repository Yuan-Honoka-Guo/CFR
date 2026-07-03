import argparse

import os
import torch
import wandb

import numpy as np
from itertools import chain

from tqdm import tqdm, trange

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big, GeGLU


def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train_CFR(args):

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shot_tag = f'_{args.few_shot}shot' if args.few_shot is not None else ''
    run_tag = f'{args.class_name}{shot_tag}'
    model_name = f'{run_tag}_{args.epochs_no}ep_{args.batch_size}bs'

    # Model saving.
    directory = f'{args.checkpoint_savepath}/{args.class_name}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    wandb.init(
        entity="thomasguoy-sichuan-university",
        project = 'crossmodal_feature_replacer',
        name = model_name
    )

    # Dataloader.
    train_loader = get_data_loader("train", class_name = args.class_name, img_size = 224, dataset_path = args.dataset_path,
                                   batch_size = args.batch_size, shuffle = True, few_shot = args.few_shot, few_shot_seed = args.few_shot_seed)
    
    # Feature extractors.
    feature_extractor = MultimodalFeatures()

    # Model instantiation.  FeatureRemappingProjection
    CFR_2Dto3D = GeGLU(in_features=768, out_features=1152, hidden_features=(768 + 1152)//2)
    CFR_3Dto2D = GeGLU(in_features=1152, out_features=768, hidden_features=(1152 + 768)//2)
    
    optimizer = torch.optim.Adam(params = chain(CFR_2Dto3D.parameters(), CFR_3Dto2D.parameters()))
    CFR_2Dto3D.to(device), CFR_3Dto2D.to(device)
    
    metric = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for epoch in trange(args.epochs_no, desc = f'Training Feature Transfer Net.'):

        epoch_cos_sim_3Dto2D, epoch_cos_sim_2Dto3D, epoch_cos_sim_2Dcycle, epoch_cos_sim_3Dcycle = [], [], [], []

        # ------------ [Trainig Loop] ------------ #
        # * Return (rgb_img, organized_pc, depth_map_3channel), globl_label
        for (rgb, pc, _), _ in tqdm(train_loader, desc = f'Extracting feature from class: {args.class_name}.'):
            rgb, pc = rgb.to(device), pc.to(device)

            # Make CFR modules trainable.
            CFR_2Dto3D.train(), CFR_3Dto2D.train()

            if args.batch_size == 1:
                rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
            else:
                rgb_patches = []
                xyz_patches = []

                for i in range(rgb.shape[0]):
                    rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb[i].unsqueeze(dim=0), pc[i].unsqueeze(dim=0))

                    rgb_patches.append(rgb_patch)
                    xyz_patches.append(xyz_patch)

                rgb_patch = torch.stack(rgb_patches, dim=0)
                xyz_patch = torch.stack(xyz_patches, dim=0)
            
            # Sample 50% of the sample, not 50% feature from each batch for both rgb_patch and xyz_patch
            # For both rgb_patch and xyz_patch, randomly sample 50% of the batch samples (not features!).
            # Make sure that this works for either batch processing (batch_size > 1) or single sample (batch_size == 1).

            # Forward Predictions.\
            rgb_feat_pred_forward = CFR_3Dto2D(xyz_patch)
            xyz_feat_pred_forward = CFR_2Dto3D(rgb_patch)

            # Backward Predictions.
            xyz_feat_pred_backward = CFR_2Dto3D(rgb_feat_pred_forward)
            rgb_feat_pred_backward = CFR_3Dto2D(xyz_feat_pred_forward)

            # Losses.
            xyz_mask = (xyz_patch.sum(axis = -1)  == 0) # Mask only the feature vectors that are 0 everywhere.
            
            # Reconstruction Loss
            loss_2Dto3D = 1 - metric(xyz_feat_pred_forward[~xyz_mask], xyz_patch[~xyz_mask]).mean()
            loss_3Dto2D = 1 - metric(rgb_feat_pred_forward[~xyz_mask], rgb_patch[~xyz_mask]).mean()
            # Self-Remapping Loss
            loss_2Dto2D = 1 - metric(rgb_feat_pred_backward[~xyz_mask], rgb_patch[~xyz_mask]).mean()
            loss_3Dto3D = 1 - metric(xyz_feat_pred_backward[~xyz_mask], xyz_patch[~xyz_mask]).mean()
            
            cos_sim_3Dto2D, cos_sim_2Dto3D = 1 - loss_3Dto2D.cpu(), 1 - loss_2Dto3D.cpu()
            cos_sim_2Dcycle, cos_sim_3Dcycle = 1 - loss_2Dto2D.cpu(), 1 - loss_3Dto3D.cpu()

            epoch_cos_sim_3Dto2D.append(cos_sim_3Dto2D), epoch_cos_sim_2Dto3D.append(cos_sim_2Dto3D)
            epoch_cos_sim_2Dcycle.append(cos_sim_2Dcycle), epoch_cos_sim_3Dcycle.append(cos_sim_3Dcycle)

            # Logging.
            wandb.log({
                "train/loss_3Dto2D" : loss_3Dto2D,
                "train/loss_2Dto3D" : loss_2Dto3D,
                "train/cosine_similarity_3Dto2D" : cos_sim_3Dto2D,
                "train/cosine_similarity_2Dto3D" : cos_sim_2Dto3D,
                "train/cosine_similarity_2Dcycle" : cos_sim_2Dcycle,
                "train/cosine_similarity_3Dcycle" : cos_sim_3Dcycle
                })

            if torch.isnan(loss_3Dto2D) or torch.isinf(loss_3Dto2D) or torch.isnan(loss_2Dto3D) or torch.isinf(loss_2Dto3D):
                print('NaN detected in loss_3Dto2D or loss_2Dto3D. Exiting...')
                exit()

            if torch.isnan(loss_2Dto2D) or torch.isinf(loss_2Dto2D) or torch.isnan(loss_3Dto3D) or torch.isinf(loss_3Dto3D):
                print('NaN detected in loss_2Dto2D or loss_3Dto3D. Exiting...')
                exit()

            # Optimization.
            if not torch.isnan(loss_3Dto2D) and not torch.isinf(loss_3Dto2D) and not torch.isnan(loss_2Dto3D) and not torch.isinf(loss_2Dto3D):
                
                optimizer.zero_grad()

                total_loss = loss_3Dto2D + loss_2Dto3D + loss_2Dto2D + loss_3Dto3D
                total_loss.backward()

                optimizer.step()

        # Global logging.
        wandb.log({
            "global_train/cos_sim_3Dto2D" : torch.Tensor(epoch_cos_sim_3Dto2D, device = 'cpu').mean(),
            "global_train/cos_sim_2Dto3D" : torch.Tensor(epoch_cos_sim_2Dto3D, device = 'cpu').mean(),
            "global_train/cos_sim_2Dcycle" : torch.Tensor(epoch_cos_sim_2Dcycle, device = 'cpu').mean(),
            "global_train/cos_sim_3Dcycle" : torch.Tensor(epoch_cos_sim_3Dcycle, device = 'cpu').mean()
            })

    torch.save(CFR_2Dto3D.state_dict(), os.path.join(directory, 'CFR_2Dto3D_' + model_name + '.pth'))
    torch.save(CFR_3Dto2D.state_dict(), os.path.join(directory, 'CFR_3Dto2D_' + model_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(directory, 'optimizer_' + model_name + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Crossmodal Feature Replacer (CFR) modules on a dataset.')

    parser.add_argument('--dataset_path', default = '/path/to/your/dataset/mvtec3d', type = str, 
                        help = 'Dataset path.')

    parser.add_argument('--checkpoint_savepath', default = '/path/to/your/checkpoints/checkpoints_CFR_mvtec', type = str, 
                        help = 'Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default = None, type = str, choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
                                                                               'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train CFR modules.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--few_shot', default = None, type = int,
                        help = 'Number of training samples to use for few-shot training.')

    parser.add_argument('--few_shot_seed', default = 42, type = int,
                        help = 'Random seed used to select few-shot samples.')
    
    args = parser.parse_args()
    train_CFR(args)
