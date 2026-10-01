#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-heldout-rust-trackA-rust-v2
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

# SFT_DATA_PROCESS.txt section 9: trackA_rust_v2 checkpoint on its matching held-out
# file only (no --no-resume: a timed-out run resumes on resubmit).
# (original note follows) sft_checkpoints/qwen3-4b-hard-v1, evaluated on the rust heldout split
# this time. See run_eval_heldout_java_hard_v1_qwen3_4b_2gpu.sh for full
# rationale; compare against run_eval_heldout_rust_baseline_qwen3_4b_2gpu.sh
# on the SAME eval_data/rl_heldout_python_rust_CLCCD.jsonl (240 q).
MODEL="sft_checkpoints/qwen3-4b-trackA-rust-v2"
QAF="eval_data/rl_heldout_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="heldout-trackA-rust-v2"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"
