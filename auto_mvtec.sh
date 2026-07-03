log_dir="${LOG_DIR:-/path/to/your/logs}"
mkdir -p "$log_dir"

for shot in 1 2 4; do
  FEW_SHOT=$shot bash 01_train_mvtec.sh
  FEW_SHOT=$shot bash 02_mb_cons_mvtec.sh
  FEW_SHOT=$shot bash 03_atten_train_mvtec.sh
  FEW_SHOT=$shot bash 04_eval_mvtec.sh 2>&1 | tee "${log_dir}/mvtec_eval_shot${shot}.log"
done
