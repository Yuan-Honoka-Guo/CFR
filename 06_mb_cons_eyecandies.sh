class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")
checkpoint_root="${CHECKPOINT_ROOT:-/path/to/your/checkpoints}"
dataset_path="${DATASET_PATH:-/path/to/your/dataset/eyecandies}"
checkpoint_savepath="${MEMORY_BANK_DIR:-${checkpoint_root}/memory_bank}"
shot="${FEW_SHOT:-4}"

for class_name in "${class_names[@]}"
    do
        python construct_memory_bank.py \
        --class_name "$class_name" \
        --dataset_path "$dataset_path" \
        --checkpoint_savepath "$checkpoint_savepath" \
        --bank_type kv \
        --few_shot "$shot"

        python construct_memory_bank.py \
        --class_name "$class_name" \
        --dataset_path "$dataset_path" \
        --checkpoint_savepath "$checkpoint_savepath" \
        --bank_type rgb \
        --few_shot "$shot"
    done
