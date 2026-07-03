epochs="${EPOCHS:-100}"
batch_size="${BATCH_SIZE:-1}"
checkpoint_root="${CHECKPOINT_ROOT:-/path/to/your/checkpoints}"
dataset_path="${DATASET_PATH:-/path/to/your/dataset/eyecandies}"
checkpoint_savepath="${CHECKPOINT_FOLDER:-${checkpoint_root}/checkpoints_CFR_eyecandies}"
memory_bank_dir="${MEMORY_BANK_DIR:-${checkpoint_root}/memory_bank}"
shot="${FEW_SHOT:-5}"

quantitative_folder="${QUANTITATIVE_FOLDER:-/path/to/your/results/quantitatives_eyecandies}"
qualitative_folder="${QUALITATIVE_FOLDER:-/path/to/your/results/qualitatives_eyecandies}"
attn_checkpoint_savepath="${ATTN_CHECKPOINT_FOLDER:-${checkpoint_root}/checkpoints_ATT_eyecandies}"

class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")

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
        --quantitative_folder "$quantitative_folder" \
        --qualitative_folder "$qualitative_folder" \
        --produce_qualitatives
    done
