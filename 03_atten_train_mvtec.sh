epochs="${EPOCHS:-100}"
batch_size="${BATCH_SIZE:-1}"
checkpoint_root="${CHECKPOINT_ROOT:-/path/to/your/checkpoints}"
dataset_path="${DATASET_PATH:-/path/to/your/dataset/mvtec3d}"
checkpoint_savepath="${ATTN_CHECKPOINT_SAVE_PATH:-${checkpoint_root}/checkpoints_ATT_mvtec}"
memory_bank_dir="${MEMORY_BANK_DIR:-${checkpoint_root}/memory_bank}"
class_names=("bagel" "cable_gland" "carrot" "cookie" "dowel" "foam" "peach" "potato" "rope" "tire")
shot="${FEW_SHOT:-4}"

for class_name in "${class_names[@]}"
    do
        python attention_training.py \
        --class_name "$class_name" \
        --few_shot "$shot" \
        --epochs_no "$epochs" \
        --batch_size "$batch_size" \
        --dataset_path "$dataset_path" \
        --checkpoint_savepath "$checkpoint_savepath" \
        --bank_path "${memory_bank_dir}/${class_name}/memory_bank_kv_${shot}shot.pt"
    done
