epochs="${EPOCHS:-100}"
batch_size="${BATCH_SIZE:-1}"
checkpoint_root="${CHECKPOINT_ROOT:-/path/to/your/checkpoints}"
dataset_path="${DATASET_PATH:-/path/to/your/dataset/eyecandies}"
checkpoint_savepath="${CHECKPOINT_SAVE_PATH:-${checkpoint_root}/checkpoints_CFR_eyecandies}"
shot="${FEW_SHOT:-4}"

class_names=("CandyCane" "ChocolateCookie" "ChocolatePraline" "Confetto" "GummyBear" "HazelnutTruffle" "LicoriceSandwich" "Lollipop" "Marshmallow" "PeppermintCandy")

for class_name in "${class_names[@]}"
    do
        python cfr_training.py \
        --class_name "$class_name" \
        --epochs_no "$epochs" \
        --batch_size "$batch_size" \
        --few_shot "$shot" \
        --dataset_path "$dataset_path" \
        --checkpoint_savepath "$checkpoint_savepath"
    done
