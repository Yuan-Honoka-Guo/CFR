import argparse
import os
from itertools import chain

import numpy as np
import torch
import wandb
from tqdm import tqdm, trange

from models.attention_retrieval import AttentionRetriever, retrieve_topk_cosine
from models.features import MultimodalFeatures
from models.dataset import get_data_loader


def set_seeds(sid=115):
    np.random.seed(sid)
    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def _load_kv_bank(bank_path, device):
    bank = torch.load(bank_path, map_location=device)
    if not isinstance(bank, dict) or "keys" not in bank or "values" not in bank:
        raise ValueError("KV memory bank not found. Build it with --bank_type kv.")
    keys = bank["keys"].to(device)
    values = bank["values"].to(device)
    return keys, values


def _sample_bank(keys, values, sample_size):
    if sample_size is None or sample_size <= 0 or sample_size >= keys.shape[0]:
        return keys, values
    idx = torch.randperm(keys.shape[0], device=keys.device)[:sample_size]
    return keys[idx], values[idx]


def _iter_features(feature_extractor, rgb, pc):
    if rgb.shape[0] == 1:
        rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
        return [rgb_patch], [xyz_patch]
    rgb_patches = []
    xyz_patches = []
    for i in range(rgb.shape[0]):
        rgb_patch, xyz_patch = feature_extractor.get_features_maps(
            rgb[i].unsqueeze(dim=0), pc[i].unsqueeze(dim=0)
        )
        rgb_patches.append(rgb_patch)
        xyz_patches.append(xyz_patch)
    return rgb_patches, xyz_patches


def train_attention(args):
    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shot_tag = f"_{args.few_shot}shot" if args.few_shot is not None else ""
    run_tag = f"{args.class_name}{shot_tag}"
    model_name = f"{run_tag}_{args.epochs_no}ep_{args.batch_size}bs"

    save_dir = os.path.join(args.checkpoint_savepath, args.class_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"ATTN_{model_name}.pth")

    wandb.init(
        entity="thomasguoy-sichuan-university",
        project="0118_rp_attention",
        name=model_name,
    )

    # Dataloader.
    train_loader = get_data_loader(
        "train",
        class_name=args.class_name,
        img_size=224,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        shuffle=True,
        few_shot=args.few_shot,
        few_shot_seed=args.few_shot_seed,
    )

    # Feature extractor.
    feature_extractor = MultimodalFeatures().to(device)
    feature_extractor.eval()

    # Load KV bank.
    bank_keys, bank_values = _load_kv_bank(args.bank_path, device=device)

    # Attention retriever.
    model = AttentionRetriever(
        q_dim=bank_keys.shape[-1],
        k_dim=bank_keys.shape[-1],
        v_dim=bank_values.shape[-1],
        d_model=args.d_model,
        dropout=args.dropout,
        normalize_qk=args.normalize_qk,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    for epoch in trange(args.epochs_no, desc="Training Attention Retriever"):
        epoch_losses = []
        for (rgb, pc, _), _ in tqdm(train_loader, desc=f"Extracting features {args.class_name}"):
            rgb, pc = rgb.to(device), pc.to(device)

            rgb_patches, xyz_patches = _iter_features(feature_extractor, rgb, pc)

            for rgb_patch, xyz_patch in zip(rgb_patches, xyz_patches):
                # Mask zero features (invalid points).
                xyz_mask = (xyz_patch.sum(axis=-1) == 0)
                valid_mask = ~xyz_mask
                if valid_mask.sum() == 0:
                    continue

                # Optionally subsample queries to control memory.
                if args.max_queries and args.max_queries > 0 and valid_mask.sum() > args.max_queries:
                    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
                    idx = valid_idx[torch.randperm(valid_idx.shape[0], device=device)[: args.max_queries]]
                    query = xyz_patch[idx]
                    target = rgb_patch[idx]
                else:
                    query = xyz_patch[valid_mask]
                    target = rgb_patch[valid_mask]

                keys, values = _sample_bank(bank_keys, bank_values, args.bank_sample_size)
                _, topk_keys, topk_values = retrieve_topk_cosine(
                    query, keys, values, k=args.bank_topk, chunk_size=args.retrieval_chunk_size
                )

                model.train()
                pred = model.forward_topk(query, topk_keys, topk_values)
                loss = 1.0 - metric(pred, target).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.detach().cpu().item()
                epoch_losses.append(loss_val)
                wandb.log({"train/loss": loss_val})

        if epoch_losses:
            epoch_mean = np.mean(epoch_losses)
            print(f"Epoch {epoch+1}/{args.epochs_no} - loss: {epoch_mean:.6f}")
            wandb.log({"train/epoch_loss": epoch_mean, "train/epoch": epoch + 1})

    torch.save(model.state_dict(), save_path)
    print(f"Saved attention model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train attention-based 3D->2D retrieval.")
    parser.add_argument("--dataset_path", default="/path/to/your/dataset/mvtec3d", type=str)
    parser.add_argument("--checkpoint_savepath", default="/path/to/your/checkpoints/checkpoints_ATT_mvtec", type=str)
    parser.add_argument("--class_name", default=None, type=str)
    parser.add_argument("--bank_path", type=str, required=True, help="Path to KV memory bank (.pt)")

    parser.add_argument("--epochs_no", default=50, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--few_shot", default=None, type=int)
    parser.add_argument("--few_shot_seed", default=42, type=int)

    parser.add_argument("--d_model", default=128, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--normalize_qk", action="store_true")
    parser.add_argument("--bank_topk", default=3, type=int, help="Top-k neighbors retrieved from the KV bank.")
    parser.add_argument("--retrieval_chunk_size", default=4096, type=int, help="Query chunk size for retrieval.")
    parser.add_argument("--bank_sample_size", default=20000, type=int, help="Randomly sample keys/values per batch.")
    parser.add_argument("--max_queries", default=20000, type=int, help="Subsample query points per image.")

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--seed", default=115, type=int)

    args = parser.parse_args()
    train_attention(args)
