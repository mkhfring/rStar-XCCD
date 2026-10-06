#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --job-name=q3-4b-codestep-smoke-eval
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Phase 0.5 smoke test (e): zero-shot code-step search on 20 CLCCD-style DEV pairs (java),
# for FORMAT checks only (train/format_check.py). Environment:
#   MODEL   model dir (models/Qwen3-4B for the base, sft_checkpoints/<name> for a smoke checkpoint)
#   BRANCH  new branch name per run (e.g. smoke-base-zeroshot, smoke-sft200)
# Submit as a twin-guard pair.

source /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/Qwen3/twin_guard.inc

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-train/rstar-xccd
: "${MODEL:?set MODEL}" "${BRANCH:?set BRANCH}"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)  MODEL=$MODEL BRANCH=$BRANCH"

module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0
export PATH=$HOME/.cargo/bin:$PATH
source ../venv-qwen3/bin/activate

START=$(date +%s)
python main.py \
  --qaf "eval_data/train/smoke_dev20_python_java_codenet.jsonl" \
  --custom_cfg "config/qwen3_4b_codestep_java_v3_zeroshot_1gpu.yaml" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"
python train/format_check.py eval_data/train/smoke_dev20_python_java_codenet_depth_16.jsonl.mcts.*."$BRANCH".*.jsonl
