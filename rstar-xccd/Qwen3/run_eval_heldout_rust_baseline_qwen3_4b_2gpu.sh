#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-heldout-rust-baseline
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

# BASELINE (un-fine-tuned) Qwen3-4B on the rust rl_heldout split ONLY
# (eval_data/rl_heldout_python_rust_CLCCD.jsonl, 240 questions). See
# run_eval_heldout_java_baseline_qwen3_4b_2gpu.sh for why this baseline
# run (rather than FINAL_FOR_CLCCD.md's full-file 0.8739 number) is what
# sft_hard_v1's rust heldout result needs to be compared against.
MODEL="models/Qwen3-4B"
QAF="eval_data/rl_heldout_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="heldout-baseline"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
