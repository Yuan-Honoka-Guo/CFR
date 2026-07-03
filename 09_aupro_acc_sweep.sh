#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash 09_aupro_acc_sweep.sh
#   DATASET=eyecandies bash 09_aupro_acc_sweep.sh
#   FEW_SHOT=4 RATE_START=1 RATE_END=50 RATE_STEP=5 bash 09_aupro_acc_sweep.sh

DATASET="${DATASET:-mvtec3d}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RATE_START="${RATE_START:-1}"
RATE_END="${RATE_END:-50}"
RATE_STEP="${RATE_STEP:-1}"
ACCEPTANCE_THRESHOLD="${ACCEPTANCE_THRESHOLD:-0.4}"
USE_ATTENTION="${USE_ATTENTION:-1}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/path/to/your/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-/path/to/your/results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/aupro_acc_sweep_${DATASET}}"

if [[ "$DATASET" == "mvtec3d" ]]; then
    FEW_SHOT="${FEW_SHOT:-4}"
    DATASET_PATH="${DATASET_PATH:-/path/to/your/dataset/mvtec3d}"
    CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-${CHECKPOINT_ROOT}/checkpoints_CFR_mvtec}"
    BANK_TEMPLATE="${BANK_TEMPLATE:-${CHECKPOINT_ROOT}/memory_bank/{class_name}/memory_bank{shot_tag}.pt}"
    ATTN_CHECKPOINT_TEMPLATE="${ATTN_CHECKPOINT_TEMPLATE:-${CHECKPOINT_ROOT}/checkpoints_ATT_mvtec/{class_name}/ATTN_{class_name}{shot_tag}_{epochs}ep_{batch_size}bs.pth}"
    ATTN_BANK_TEMPLATE="${ATTN_BANK_TEMPLATE:-${CHECKPOINT_ROOT}/memory_bank/{class_name}/memory_bank_kv{shot_tag}.pt}"
elif [[ "$DATASET" == "eyecandies" ]]; then
    FEW_SHOT="${FEW_SHOT:-5}"
    DATASET_PATH="${DATASET_PATH:-/path/to/your/dataset/eyecandies}"
    CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-${CHECKPOINT_ROOT}/checkpoints_CFR_eyecandies}"
    BANK_TEMPLATE="${BANK_TEMPLATE:-${CHECKPOINT_ROOT}/memory_bank/{class_name}/memory_bank{shot_tag}.pt}"
    ATTN_CHECKPOINT_TEMPLATE="${ATTN_CHECKPOINT_TEMPLATE:-${CHECKPOINT_ROOT}/checkpoints_ATT_eyecandies/{class_name}/ATTN_{class_name}{shot_tag}_{epochs}ep_{batch_size}bs.pth}"
    ATTN_BANK_TEMPLATE="${ATTN_BANK_TEMPLATE:-${CHECKPOINT_ROOT}/memory_bank/{class_name}/memory_bank_kv{shot_tag}.pt}"
else
    echo "Unsupported DATASET=$DATASET. Use mvtec3d or eyecandies." >&2
    exit 1
fi

cmd=(
    python aupro_acc_sweep.py
    --dataset_name "$DATASET"
    --dataset_path "$DATASET_PATH"
    --checkpoint_folder "$CHECKPOINT_FOLDER"
    --output_dir "$OUTPUT_DIR"
    --epochs_no "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --few_shot "$FEW_SHOT"
    --acceptance_threshold "$ACCEPTANCE_THRESHOLD"
    --rate_start_percent "$RATE_START"
    --rate_end_percent "$RATE_END"
    --rate_step_percent "$RATE_STEP"
    --bank_path_template "$BANK_TEMPLATE"
    --attn_checkpoint_template "$ATTN_CHECKPOINT_TEMPLATE"
    --attn_bank_path_template "$ATTN_BANK_TEMPLATE"
)

if [[ "$USE_ATTENTION" == "1" ]]; then
    cmd+=(--use_attention_retrieval)
else
    cmd+=(--no-use_attention_retrieval)
fi

"${cmd[@]}"
