log_dir="${LOG_DIR:-/path/to/your/logs}"
mkdir -p "$log_dir"

for shot in 1 2 4; do
  FEW_SHOT=$shot bash 05_train_eyecandies.sh
  FEW_SHOT=$shot bash 06_mb_cons_eyecandies.sh
  FEW_SHOT=$shot bash 07_atten_train_eyecandies.sh
  FEW_SHOT=$shot bash 08_eval_eyecandies.sh 2>&1 | tee "${log_dir}/eyecandies_eval_shot${shot}.log"
done
