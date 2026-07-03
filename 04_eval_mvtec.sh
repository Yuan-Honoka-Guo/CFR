epochs="${EPOCHS:-100}"
batch_size="${BATCH_SIZE:-1}"
checkpoint_root="${CHECKPOINT_ROOT:-/path/to/your/checkpoints}"
dataset_path="${DATASET_PATH:-/path/to/your/dataset/mvtec3d}"
checkpoint_savepath="${CHECKPOINT_FOLDER:-${checkpoint_root}/checkpoints_CFR_mvtec}"
memory_bank_dir="${MEMORY_BANK_DIR:-${checkpoint_root}/memory_bank}"
attn_checkpoint_savepath="${ATTN_CHECKPOINT_FOLDER:-${checkpoint_root}/checkpoints_ATT_mvtec}"
shot="${FEW_SHOT:-4}"

class_names=("bagel" "cable_gland" "carrot" "cookie" "dowel" "foam" "peach" "potato" "rope" "tire")

for class_name in "${class_names[@]}"
    do
        python cfr_inference.py \
        --class_name "$class_name" \
        --epochs_no "$epochs" \
        --batch_size "$batch_size" \
        --few_shot "$shot" \
        --dataset_path "$dataset_path" \
        --checkpoint_folder "$checkpoint_savepath" \
        --bank_path "${memory_bank_dir}/${class_name}/memory_bank_${shot}shot.pt" \
        --attn_checkpoint "${attn_checkpoint_savepath}/${class_name}/ATTN_${class_name}_${shot}shot_${epochs}ep_${batch_size}bs.pth" \
        --attn_bank_path "${memory_bank_dir}/${class_name}/memory_bank_kv_${shot}shot.pt" \
        --acceptance_threshold 0.4
    done
