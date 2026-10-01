#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-heldout-java-hard-java-v1
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

# sft_checkpoints/qwen3-4b-hard-java-v1 (jobs 3645951/3645952, 2026-09-22):
# fine-tuned on 40 safe leaves from ONLY the java hard-mining track
# (eval_data/sft_hard_v1_java.jsonl -- see
# Qwen3/run_sft_hard_java_v1_qwen3_4b_4gpu_fard.sh). Evaluated only on
# java heldout (this is a java-only model; no rust eval, nothing in its
# training data could plausibly transfer). Compare against
# run_eval_heldout_java_baseline_qwen3_4b_2gpu.sh AND
# run_eval_heldout_java_hard_v1_qwen3_4b_2gpu.sh on this SAME 360-question
# file -- three-way comparison: base model, java-only fine-tune,
# java+rust combined fine-tune.
MODEL="sft_checkpoints/qwen3-4b-hard-java-v1"
QAF="eval_data/rl_heldout_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="heldout-hard-java-v1"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
