# Crossmodal Feature Replacer (CFR)

Crossmodal Feature Replacer (CFR) is a multimodal anomaly detection model for 2D-3D industrial anomaly detection.

The model improves crossmodal reconstruction reliability by replacing unreliable reconstructed features with better matches from memory banks.

CFR is built upon Crossmodal Feature Mapping (CFM), which serves as the baseline model, and inherits its preprocessing pipeline and feature extraction settings. Based on this foundation, CFR further introduces:

- Bidirectional cyclic reconstruction
- Modality-specific and paired memory banks
- Optional attention-based replacement
- Dynamic feature replacement during inference

This repository is released for research reproducibility and experimental reference.

---

# Overview

The complete CFR pipeline consists of four stages:

1. Train the CFR reconstruction modules
2. Construct memory banks
3. Train the attention retriever
4. Perform evaluation and visualization

The provided shell scripts already organize the full workflow and can be executed directly.

---

# Repository Structure

```text
.
├── cfr_training.py
├── cfr_inference.py
├── construct_memory_bank.py
├── attention_training.py
│
├── 01_train_mvtec.sh
├── 02_mb_cons_mvtec.sh
├── 03_atten_train_mvtec.sh
├── 04_eval_mvtec.sh
│
├── 05_train_eyecandies.sh
├── 06_mb_cons_eyecandies.sh
├── 07_atten_train_eyecandies.sh
├── 08_eval_eyecandies.sh
│
├── auto_mvtec.sh
├── auto_eyecandies.sh
└── runner.sh
```

---

# Environment Setup

## Tested Environment

- Python 3.8+
- CUDA 11.x
- Ubuntu 20.04
- PyTorch 1.12+

CUDA is strongly recommended for both training and inference.

---

## Install Dependencies

```bash
pip install torch torchvision numpy scipy scikit-learn Pillow matplotlib tqdm timm
```

You also need:

```bash
pointnet2_ops
```

Please compile/install it following the official PointNet++ implementation.

---

# Pretrained Feature Extractors

CFR uses the same pretrained feature extractors as CFM:

- DINO for RGB features
- PointMAE for point cloud features

Expected directory structure:

```text
/path/to/your/FeatureExtractorWeights/
├── dino_vitbase8_pretrain.pth
└── pointmae_pretrain.pth
```

By default, the repository uses:

```text
/path/to/your/FeatureExtractorWeights/dino_vitbase8_pretrain.pth
/path/to/your/FeatureExtractorWeights/pointmae_pretrain.pth
```

You can override this without editing code:

```bash
export FEATURE_EXTRACTOR_WEIGHTS=/path/to/your/FeatureExtractorWeights
```

---

# Dataset Preparation

## Supported Datasets

- MVTec 3D-AD
- Eyecandies

---

# Preprocessing Follows CFM

The preprocessing pipeline used in this repository follows the official implementation of Crossmodal Feature Mapping (CFM).

Please first preprocess the datasets following the CFM repository:

https://github.com/CVLAB-Unibo/crossmodal-feature-mapping

The preprocessing procedure, feature organization, and data format are inherited from CFM.

---

# Recommended Dataset Structure

After preprocessing, the dataset directory should look similar to:

```text
datasets/
├── mvtec_3d/
│   ├── bagel/
│   │   ├── train/
│   │   ├── test/
│   │   ├── rgb/
│   │   ├── xyz/
│   │   ├── gt/
│   │   └── ...
│   │
│   ├── cable_gland/
│   ├── carrot/
│   └── ...
```

Please ensure:

- RGB images are correctly generated
- Point cloud / XYZ files are correctly processed
- The directory hierarchy matches the CFM preprocessing output

---

# Configure Dataset Path

Example:

```bash
DATASET_PATH=/path/to/your/dataset/mvtec3d \
CHECKPOINT_ROOT=/path/to/your/checkpoints \
MEMORY_BANK_DIR=/path/to/your/checkpoints/memory_bank \
FEW_SHOT=4 \
bash 01_train_mvtec.sh
```

The shell scripts read paths from environment variables. Use `/path/to/your/...` as a placeholder for your local dataset, checkpoint, result, and log directories.

---

# Full Training Pipeline (MVTec 3D-AD)

## Step 1 — Train CFR

```bash
bash 01_train_mvtec.sh
```

This stage trains the bidirectional crossmodal reconstruction modules.

Generated checkpoints will be saved to:

```text
/path/to/your/checkpoints/checkpoints_CFR_mvtec/
```

---

## Step 2 — Construct Memory Banks

```bash
bash 02_mb_cons_mvtec.sh
```

Two types of memory banks are constructed:

### RGB Memory Bank

Used for feature replacement and distance-based anomaly scoring.

### KV Replacement

Used for optional attention-based replacement.

---

## Step 3 — Train Attention Retriever

```bash
bash 03_atten_train_mvtec.sh
```

This stage trains the attention retriever using the KV replacement bank.

Generated checkpoints will be saved to:

```text
/path/to/your/checkpoints/checkpoints_ATT_mvtec/
```

---

## Step 4 — Evaluation

```bash
bash 04_eval_mvtec.sh
```

This stage performs:

- anomaly detection
- feature replacement
- metric evaluation
- qualitative visualization generation

Results will be saved under:

```text
/path/to/your/results/
```

---

# Full Training Pipeline (Eyecandies)

```bash
bash 05_train_eyecandies.sh
bash 06_mb_cons_eyecandies.sh
bash 07_atten_train_eyecandies.sh
bash 08_eval_eyecandies.sh
```

---

# Automatic End-to-End Pipeline

## MVTec 3D-AD

```bash
bash auto_mvtec.sh
```

## Eyecandies

```bash
bash auto_eyecandies.sh
```

These scripts sequentially execute:

1. CFR training
2. Memory bank construction
3. Attention retriever training
4. Evaluation

---

# Direct Python Usage

## Train CFR

```bash
python cfr_training.py \
  --class_name bagel \
  --dataset_path /path/to/your/dataset/mvtec3d \
  --checkpoint_savepath /path/to/your/checkpoints/checkpoints_CFR_mvtec
```

---

## Construct Memory Bank

```bash
python construct_memory_bank.py \
  --class_name bagel \
  --dataset_path /path/to/your/dataset/mvtec3d \
  --checkpoint_savepath /path/to/your/checkpoints/memory_bank \
  --bank_type kv \
  --few_shot 4
```

### Supported Memory Bank Types

| Type | Description |
|------|-------------|
| rgb  | RGB replacement bank |
| kv   | KV replacement bank |

---

## Train Attention Retriever

```bash
python attention_training.py \
  --class_name bagel \
  --dataset_path /path/to/your/dataset/mvtec3d \
  --checkpoint_savepath /path/to/your/checkpoints/checkpoints_ATT_mvtec \
  --bank_path /path/to/your/checkpoints/memory_bank/bagel/memory_bank_kv_4shot.pt
```

---

## Inference

```bash
python cfr_inference.py \
  --class_name bagel \
  --dataset_path /path/to/your/dataset/mvtec3d \
  --checkpoint_folder /path/to/your/checkpoints/checkpoints_CFR_mvtec \
  --bank_path /path/to/your/checkpoints/memory_bank/bagel/memory_bank_4shot.pt \
  --attn_checkpoint /path/to/your/checkpoints/checkpoints_ATT_mvtec/bagel/ATTN_bagel_4shot_100ep_1bs.pth \
  --attn_bank_path /path/to/your/checkpoints/memory_bank/bagel/memory_bank_kv_4shot.pt
```

---

# Output Structure

```text
/path/to/your/results/
├── quantitatives_mvtec/
├── qualitatives_mvtec/
├── quantitatives_eyecandies/
└── qualitatives_eyecandies/
```

---

# Notes

- CUDA is strongly recommended
- Incorrect dataset preprocessing is the most common source of runtime errors
- Please ensure the pretrained weights are correctly placed before training
- Few-shot settings can be controlled through:

```bash
FEW_SHOT=4
```

---

# Acknowledgements

This repository is built upon the official implementations of:

- Crossmodal Feature Mapping (CFM)
- PatchCore

In particular, the dataset preprocessing pipeline and feature extraction settings follow the official CFM implementation.

Please cite the original works if you use this repository in your research.
