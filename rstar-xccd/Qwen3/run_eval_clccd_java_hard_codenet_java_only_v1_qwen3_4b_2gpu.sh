#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-java-hard-codenet-java-only-v1
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

# sft_checkpoints/qwen3-4b-hard-codenet-java-only-v1 (job 3880506, 2026-09-24):
# fine-tuned on 58 safe leaves (23 unique questions) mined from the "hard"
# subset of eval_data/codenet_train_python_java.jsonl -- the 63/2000
# questions Qwen3-4B's own aggregate MCTS verdict got wrong (see
# SFT_DATA_PROCESS.txt section 8), re-mined with is_sampling=True on branch
# rl-mining-hard-codenet-v1 (jobs 3870570/3870571). Source question pool is
# eval_data/codenet_train_python_java.jsonl, the same Track-B CodeNet pool
# codenet-all-v1 was built from -- disjoint from the CLCCD test files (see
# SFT_DATA_PROCESS.txt section 2), so this checkpoint can be evaluated on
# the FULL CLCCD test set directly, same as codenet-all-v1 /
# run_eval_clccd_java_codenet_all_v1_qwen3_4b_2gpu.sh. Same eval config as
# the published Qwen3-4B baseline (job 2429953, F1 0.9454 -- FINAL_FOR_CLCCD.md)
# and as codenet-all-v1's own eval, so directly comparable to both.
#
# TIME BUDGET: matches codenet-all-v1's java eval budget -- base model took
# 8h13m on these 1800 questions, fine-tuned checkpoints have shown 3-8x
# slower inference (fp32 HF save from train_SFT.py rather than bf16).
# Budgeted well above the worst-case extrapolation to avoid a timeout.
MODEL="sft_checkpoints/qwen3-4b-hard-codenet-java-only-v1"
QAF="eval_data/test_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="assert-consistency-score"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
