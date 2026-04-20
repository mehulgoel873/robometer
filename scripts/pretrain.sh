#!/bin/bash
#SBATCH --partition=general
#SBATCH --job-name=rbm_pretrain
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --chdir=/home/mehulg/robometer
#SBATCH --output=/home/mehulg/robometer/logs/rbm_pretrain_%j.out
#SBATCH --error=/home/mehulg/robometer/logs/rbm_pretrain_%j.out
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

mkdir -p logs
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"
set -euo pipefail
export PYTHONUNBUFFERED=1

# === Run ===
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/train.py" ] && [ -d "$SLURM_SUBMIT_DIR/robometer" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
elif [ -f "$PWD/train.py" ] && [ -d "$PWD/robometer" ]; then
    PROJECT_ROOT="$PWD"
elif [ -f "/home/mehulg/robometer/train.py" ] && [ -d "/home/mehulg/robometer/robometer" ]; then
    PROJECT_ROOT="/home/mehulg/robometer"
else
    echo "Could not determine project root"
    exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "Project root: $PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found in PATH"
    exit 1
fi

export ROBOMETER_PROCESSED_DATASETS_PATH=/data/user_data/mehulg/robometer_processed/


uv run python train.py \
  model.base_model_id=Qwen/Qwen3-VL-4B-Instruct \
  model.use_peft=true \
  model.train_progress_head=true \
  model.train_preference_head=true \
  data.train_datasets=[mehulgoel_robomimic_rbm_train_robomimic] \
  data.eval_datasets=[mehulgoel_robomimic_rbm_eval_robomimic_eval] \
  training.load_from_checkpoint=robometer/Robometer-4B \
  training.per_device_train_batch_size=8 \
  training.learning_rate=2e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.max_steps=1000 \
  training.output_dir=/data/user_data/mehulg/robometer/checkpoints/ \
  training.exp_name=robometer4b_lora_robomimic \
  logging.log_to=[wandb] \
  logging.wandb_entity=pluralistic-goal-conditioning \
  logging.wandb_project=robometer \
  custom_eval.eval_types=[reward_alignment] \
  custom_eval.reward_alignment=[mehulgoel_robomimic_rbm_eval_robomimic_eval] \
  logging.save_best.metric_names=[eval_rew_align/pearson_mehulgoel_robomimic_rbm_eval_robomimic_eval] \
  logging.save_best.greater_is_better=[true] \
  training.overwrite_output_dir=True \
  training.eval_steps=1 \
  training.custom_eval_steps=1
