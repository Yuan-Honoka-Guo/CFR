import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from cfr_inference import _load_kv_bank, _sample_kv_bank, set_seeds
from models.attention_retrieval import AttentionRetriever, retrieve_topk_cosine
from models.dataset import eyecandies_classes, get_data_loader, mvtec3d_classes
from models.feature_transfer_nets import GeGLU
from models.features import MultimodalFeatures
from test_memory_bank import MemoryBankWrapper
from utils.metrics_utils import calculate_au_pro


def parse_rates(rate_string):
    rates = []
    for raw_rate in rate_string.split(","):
        raw_rate = raw_rate.strip()
        if not raw_rate:
            continue
        value = float(raw_rate)
        if value > 1.0:
            value /= 100.0
        if value <= 0.0 or value >= 1.0:
            raise ValueError(f"Rate must be in (0, 1), got {raw_rate}.")
        rates.append(value)
    if not rates:
        raise ValueError("At least one AUPRO rate is required.")
    return rates


def rates_from_range(start_percent, end_percent, step_percent):
    if start_percent <= 0 or end_percent <= 0 or step_percent <= 0:
        raise ValueError("Rate range values must be positive.")
    if start_percent > end_percent:
        raise ValueError("--rate_start_percent must be <= --rate_end_percent.")

    rates = []
    current = start_percent
    # Keep integer-like decimal ranges stable, e.g. 1,2,...,50.
    while current <= end_percent + 1e-9:
        rates.append(current / 100.0)
        current += step_percent
    return rates


def get_class_names(dataset_name):
    if dataset_name == "mvtec3d":
        return mvtec3d_classes()
    if dataset_name == "eyecandies":
        return eyecandies_classes()
    if dataset_name == "all":
        return mvtec3d_classes() + eyecandies_classes()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def make_shot_tag(few_shot):
    return f"_{few_shot}shot" if few_shot is not None else ""


def build_class_paths(args, class_name):
    shot_tag = make_shot_tag(args.few_shot)
    bank_shot_tag = f"_{args.few_shot}shot" if args.few_shot is not None else "_full"
    model_name = f"{class_name}{shot_tag}_{args.epochs_no}ep_{args.batch_size}bs"

    paths = {
        "bank_path": args.bank_path_template.format(
            class_name=class_name,
            shot=args.few_shot,
            shot_tag=bank_shot_tag,
            epochs=args.epochs_no,
            batch_size=args.batch_size,
        ),
        "attn_bank_path": args.attn_bank_path_template.format(
            class_name=class_name,
            shot=args.few_shot,
            shot_tag=bank_shot_tag,
            epochs=args.epochs_no,
            batch_size=args.batch_size,
        ),
        "attn_checkpoint": args.attn_checkpoint_template.format(
            class_name=class_name,
            shot=args.few_shot,
            shot_tag=shot_tag,
            epochs=args.epochs_no,
            batch_size=args.batch_size,
        ),
        "cfr_2dto3d": os.path.join(
            args.checkpoint_folder,
            class_name,
            f"CFR_2Dto3D_{model_name}.pth",
        ),
        "cfr_3dto2d": os.path.join(
            args.checkpoint_folder,
            class_name,
            f"CFR_3Dto2D_{model_name}.pth",
        ),
    }
    return paths


def ensure_required_paths(paths, use_attention_retrieval):
    required = ["bank_path", "cfr_2dto3d", "cfr_3dto2d"]
    if use_attention_retrieval:
        required.extend(["attn_bank_path", "attn_checkpoint"])
    missing = [path for key, path in paths.items() if key in required and not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))


def safe_normalize_map(score_map):
    nonzero = score_map[score_map != 0]
    if nonzero.numel() == 0:
        return score_map
    return score_map / nonzero.mean().clamp_min(1e-12)


def safe_image_score_map(score_map):
    nonzero = score_map[score_map != 0]
    if nonzero.numel() == 0:
        return score_map
    return score_map / torch.sqrt(nonzero.mean().clamp_min(1e-12))


def safe_pixel_score_map(score_map):
    return score_map / torch.sqrt(score_map.mean().clamp_min(1e-12))


def collect_predictions(args, class_name, paths, device):
    test_loader = get_data_loader(
        "test",
        class_name=class_name,
        img_size=224,
        dataset_path=args.dataset_path,
        batch_size=1,
        shuffle=False,
    )

    feature_extractor = MultimodalFeatures()

    cfr_2dto3d = GeGLU(in_features=768, out_features=1152, hidden_features=(768 + 1152) // 2)
    cfr_3dto2d = GeGLU(in_features=1152, out_features=768, hidden_features=(1152 + 768) // 2)
    cfr_2dto3d.load_state_dict(torch.load(paths["cfr_2dto3d"], map_location=device))
    cfr_3dto2d.load_state_dict(torch.load(paths["cfr_3dto2d"], map_location=device))
    cfr_2dto3d.to(device).eval()
    cfr_3dto2d.to(device).eval()

    memory_bank = MemoryBankWrapper(paths["bank_path"], device=device)

    attn_model = None
    kv_keys, kv_values = None, None
    if args.use_attention_retrieval:
        kv_keys, kv_values = _load_kv_bank(paths["attn_bank_path"], device=device)
        attn_model = AttentionRetriever(
            q_dim=kv_keys.shape[-1],
            k_dim=kv_keys.shape[-1],
            v_dim=kv_values.shape[-1],
            d_model=args.attn_d_model,
            dropout=0.0,
            normalize_qk=args.attn_normalize_qk,
        ).to(device)
        attn_model.load_state_dict(torch.load(paths["attn_checkpoint"], map_location=device))
        attn_model.eval()

    weight_l = torch.ones(1, 1, 5, 5, device=device) / 25

    gts = []
    predictions = []
    image_labels = []
    image_preds = []
    pixel_labels = []
    pixel_preds = []

    with torch.no_grad():
        for (rgb, pc, _), gt, label, _ in tqdm(
            test_loader,
            desc=f"Sweeping {class_name}",
            leave=False,
        ):
            rgb, pc = rgb.to(device), pc.to(device)
            rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)

            rgb_feat_pred = cfr_3dto2d(xyz_patch)
            xyz_feat_pred = cfr_2dto3d(rgb_patch)

            flat_pred = rgb_feat_pred.reshape(-1, rgb_feat_pred.shape[-1])
            flat_real = rgb_patch.reshape(-1, rgb_patch.shape[-1])
            flat_xyz = xyz_patch.reshape(-1, xyz_patch.shape[-1])
            xyz_mask_flat = flat_xyz.sum(axis=-1) == 0

            dists_pred, _ = memory_bank.get_nearest_neighbors_consine(flat_pred, k=1)
            min_dists = dists_pred[:, 0]
            valid_replace = (min_dists >= args.acceptance_threshold) & (~xyz_mask_flat)
            if valid_replace.sum() > 0:
                if args.use_attention_retrieval:
                    keys, values = _sample_kv_bank(kv_keys, kv_values, args.attn_bank_sample_size)
                    query = flat_xyz[valid_replace]
                    _, topk_keys, topk_values = retrieve_topk_cosine(
                        query,
                        keys,
                        values,
                        k=args.attn_retrieval_topk,
                        chunk_size=args.attn_chunk_size,
                    )
                    flat_pred[valid_replace] = attn_model.forward_topk(query, topk_keys, topk_values)
                else:
                    feats_to_query = flat_real[valid_replace]
                    _, feats_replacement = memory_bank.get_nearest_neighbors_consine(feats_to_query, k=1)
                    flat_pred[valid_replace] = feats_replacement.squeeze(1)
                rgb_feat_pred = flat_pred.reshape(rgb_feat_pred.shape)

            xyz_mask = xyz_patch.sum(axis=-1) == 0
            cos_3d = (
                torch.nn.functional.normalize(xyz_feat_pred, dim=1)
                - torch.nn.functional.normalize(xyz_patch, dim=1)
            ).pow(2).sum(1).sqrt()
            cos_3d[xyz_mask] = 0.0
            cos_3d = cos_3d.reshape(224, 224)

            cos_2d = (
                torch.nn.functional.normalize(rgb_feat_pred, dim=1)
                - torch.nn.functional.normalize(rgb_patch, dim=1)
            ).pow(2).sum(1).sqrt()
            cos_2d[xyz_mask] = 0.0
            cos_2d = cos_2d.reshape(224, 224)

            cos_comb = cos_2d * cos_3d
            cos_comb.reshape(-1)[xyz_mask] = 0.0
            cos_comb = cos_comb.reshape(1, 1, 224, 224)
            cos_comb = torch.nn.functional.conv2d(cos_comb, padding=2, weight=weight_l)
            cos_comb = cos_comb.reshape(224, 224)

            anomaly_map = safe_normalize_map(cos_comb).cpu().numpy()
            image_score_map = safe_image_score_map(cos_comb).cpu().numpy()
            pixel_score_map = safe_pixel_score_map(cos_comb).cpu().numpy()
            gt_map = gt.squeeze().cpu().numpy()

            gts.append(gt_map)
            predictions.append(anomaly_map)
            image_labels.append(int(label.item() if torch.is_tensor(label) else label))
            image_preds.append(float(image_score_map.max()))
            pixel_labels.extend(gt_map.reshape(-1).astype(np.int32).tolist())
            pixel_preds.extend(pixel_score_map.reshape(-1).tolist())

    return {
        "gts": gts,
        "predictions": predictions,
        "image_labels": np.asarray(image_labels, dtype=np.int32),
        "image_preds": np.asarray(image_preds, dtype=np.float64),
        "pixel_labels": np.asarray(pixel_labels, dtype=np.int32),
        "pixel_preds": np.asarray(pixel_preds, dtype=np.float64),
    }


def threshold_from_normal_scores(scores, labels, target_fpr):
    normal_scores = np.asarray(scores)[np.asarray(labels) == 0]
    if normal_scores.size == 0:
        raise ValueError("Cannot choose an FPR threshold without normal samples/pixels.")
    normal_scores = np.sort(normal_scores)
    allowed_false_positives = int(np.floor(target_fpr * normal_scores.size))
    threshold_index = max(0, normal_scores.size - allowed_false_positives - 1)
    return float(normal_scores[threshold_index])


def binary_metrics(scores, labels, threshold):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(np.int32)
    preds = (scores > threshold).astype(np.int32)
    acc = float((preds == labels).mean())
    balanced_acc = float(balanced_accuracy_score(labels, preds))

    normal_mask = labels == 0
    anomaly_mask = labels == 1
    fpr = float(preds[normal_mask].mean()) if normal_mask.any() else float("nan")
    tpr = float(preds[anomaly_mask].mean()) if anomaly_mask.any() else float("nan")
    return acc, balanced_acc, fpr, tpr


def compute_rate_rows(class_name, data, rates):
    au_pros, _ = calculate_au_pro(data["gts"], data["predictions"], integration_limit=rates)

    rows = []
    for rate, au_pro in zip(rates, au_pros):
        image_threshold = threshold_from_normal_scores(
            data["image_preds"],
            data["image_labels"],
            rate,
        )
        image_acc, image_balanced_acc, image_fpr, image_tpr = binary_metrics(
            data["image_preds"],
            data["image_labels"],
            image_threshold,
        )

        pixel_threshold = threshold_from_normal_scores(
            data["pixel_preds"],
            data["pixel_labels"],
            rate,
        )
        pixel_acc, pixel_balanced_acc, pixel_fpr, pixel_tpr = binary_metrics(
            data["pixel_preds"],
            data["pixel_labels"],
            pixel_threshold,
        )

        rows.append(
            {
                "class_name": class_name,
                "rate_percent": rate * 100.0,
                "aupro": float(au_pro),
                "image_acc": image_acc,
                "image_balanced_acc": image_balanced_acc,
                "image_threshold": image_threshold,
                "image_fpr": image_fpr,
                "image_tpr": image_tpr,
                "pixel_acc": pixel_acc,
                "pixel_balanced_acc": pixel_balanced_acc,
                "pixel_threshold": pixel_threshold,
                "pixel_fpr": pixel_fpr,
                "pixel_tpr": pixel_tpr,
            }
        )
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "class_name",
        "rate_percent",
        "aupro",
        "image_acc",
        "image_balanced_acc",
        "image_threshold",
        "image_fpr",
        "image_tpr",
        "pixel_acc",
        "pixel_balanced_acc",
        "pixel_threshold",
        "pixel_fpr",
        "pixel_tpr",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows, rates):
    summary = []
    for rate in rates:
        rate_rows = [row for row in rows if abs(row["rate_percent"] - rate * 100.0) < 1e-9]
        summary.append(
            {
                "class_name": "mean",
                "rate_percent": rate * 100.0,
                "aupro": float(np.mean([row["aupro"] for row in rate_rows])),
                "image_acc": float(np.mean([row["image_acc"] for row in rate_rows])),
                "image_balanced_acc": float(np.mean([row["image_balanced_acc"] for row in rate_rows])),
                "image_threshold": float("nan"),
                "image_fpr": float(np.mean([row["image_fpr"] for row in rate_rows])),
                "image_tpr": float(np.mean([row["image_tpr"] for row in rate_rows])),
                "pixel_acc": float(np.mean([row["pixel_acc"] for row in rate_rows])),
                "pixel_balanced_acc": float(np.mean([row["pixel_balanced_acc"] for row in rate_rows])),
                "pixel_threshold": float("nan"),
                "pixel_fpr": float(np.mean([row["pixel_fpr"] for row in rate_rows])),
                "pixel_tpr": float(np.mean([row["pixel_tpr"] for row in rate_rows])),
            }
        )
    return summary


def plot_curve(path, rows, value_key, ylabel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    class_names = sorted({row["class_name"] for row in rows})
    plt.figure(figsize=(10, 6))
    for class_name in class_names:
        class_rows = sorted(
            [row for row in rows if row["class_name"] == class_name],
            key=lambda item: item["rate_percent"],
        )
        plt.plot(
            [row["rate_percent"] for row in class_rows],
            [row[value_key] for row in class_rows],
            marker="o",
            linewidth=1.2,
            markersize=2.5,
            label=class_name,
        )
    plt.xlabel("AUPRO integration limit / target FPR (%)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sweep AUPRO integration rates and report accuracy curves for all classes."
    )
    parser.add_argument("--dataset_name", default="mvtec3d", choices=["mvtec3d", "eyecandies", "all"])
    parser.add_argument("--dataset_path", default="/path/to/your/dataset/mvtec3d", type=str)
    parser.add_argument("--checkpoint_folder", default="/path/to/your/checkpoints/checkpoints_CFR_mvtec", type=str)
    parser.add_argument("--output_dir", default="/path/to/your/results/aupro_acc_sweep", type=str)
    parser.add_argument("--class_names", nargs="*", default=None)

    parser.add_argument("--epochs_no", default=100, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--few_shot", default=None, type=int)
    parser.add_argument("--acceptance_threshold", default=0.4, type=float)

    parser.add_argument("--rate_start_percent", default=1.0, type=float)
    parser.add_argument("--rate_end_percent", default=50.0, type=float)
    parser.add_argument("--rate_step_percent", default=1.0, type=float)
    parser.add_argument(
        "--rates",
        default=None,
        type=str,
        help="Comma-separated rates. Values >1 are treated as percentages, e.g. 1,5,10,50.",
    )

    parser.add_argument(
        "--bank_path_template",
        default="/path/to/your/checkpoints/memory_bank/{class_name}/memory_bank{shot_tag}.pt",
        type=str,
        help="Format fields: {class_name}, {shot}, {shot_tag}, {epochs}, {batch_size}.",
    )
    parser.add_argument("--use_attention_retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--attn_checkpoint_template",
        default="/path/to/your/checkpoints/checkpoints_ATT_mvtec/{class_name}/ATTN_{class_name}{shot_tag}_{epochs}ep_{batch_size}bs.pth",
        type=str,
    )
    parser.add_argument(
        "--attn_bank_path_template",
        default="/path/to/your/checkpoints/memory_bank/{class_name}/memory_bank_kv{shot_tag}.pt",
        type=str,
    )
    parser.add_argument("--attn_d_model", default=128, type=int)
    parser.add_argument("--attn_normalize_qk", default=False, action="store_true")
    parser.add_argument("--attn_chunk_size", default=4096, type=int)
    parser.add_argument("--attn_bank_sample_size", default=20000, type=int)
    parser.add_argument("--attn_retrieval_topk", default=3, type=int)

    args = parser.parse_args()
    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rates = parse_rates(args.rates) if args.rates else rates_from_range(
        args.rate_start_percent,
        args.rate_end_percent,
        args.rate_step_percent,
    )
    class_names = args.class_names if args.class_names else get_class_names(args.dataset_name)

    all_rows = []
    for class_name in class_names:
        paths = build_class_paths(args, class_name)
        ensure_required_paths(paths, args.use_attention_retrieval)
        data = collect_predictions(args, class_name, paths, device)
        class_rows = compute_rate_rows(class_name, data, rates)
        all_rows.extend(class_rows)

        class_csv = os.path.join(args.output_dir, f"{class_name}_aupro_acc_sweep.csv")
        write_csv(class_csv, class_rows)

        image_auc = roc_auc_score(data["image_labels"], data["image_preds"])
        pixel_auc = roc_auc_score(data["pixel_labels"], data["pixel_preds"])
        print(f"{class_name}: I-AUROC={image_auc:.4f}, P-AUROC={pixel_auc:.4f}, wrote {class_csv}")

    all_csv = os.path.join(args.output_dir, "all_classes_aupro_acc_sweep.csv")
    summary_csv = os.path.join(args.output_dir, "mean_aupro_acc_sweep.csv")
    write_csv(all_csv, all_rows)
    write_csv(summary_csv, summarize_rows(all_rows, rates))

    plot_curve(
        os.path.join(args.output_dir, "image_acc_vs_aupro_rate.png"),
        all_rows + summarize_rows(all_rows, rates),
        "image_acc",
        "Image accuracy",
    )
    plot_curve(
        os.path.join(args.output_dir, "image_balanced_acc_vs_aupro_rate.png"),
        all_rows + summarize_rows(all_rows, rates),
        "image_balanced_acc",
        "Image balanced accuracy",
    )
    plot_curve(
        os.path.join(args.output_dir, "aupro_vs_rate.png"),
        all_rows + summarize_rows(all_rows, rates),
        "aupro",
        "AUPRO",
    )
    print(f"Wrote combined results to {all_csv}")
    print(f"Wrote mean results to {summary_csv}")


if __name__ == "__main__":
    main()
