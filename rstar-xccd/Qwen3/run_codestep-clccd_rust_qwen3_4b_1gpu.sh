#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Twin guard (both GPU accounts; see twin_guard.inc)
source /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/Qwen3/twin_guard.inc

# CODE-STEP on the FULL CLCCD TEST (2026-10-03, user go-ahead after the pilots): prompt v2 for rust
# (java v3; rust v2 because v3 lost rust precision on pilot2 -- rust v3 to be tested later).
# Seed-0 pilot pairs (codestep_pilot/pilot_*) were DEV data for v1: exclude or flag them in the report.
# Compare against SCB (ablation-pyonly) and the frozen extension (clccd-ext-it8v3) in the main tree. Resume: resubmit unchanged.
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-codestep/rstar-xccd

echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0
export PATH=$HOME/.cargo/bin:$PATH
source ../venv-qwen3/bin/activate

START=$(date +%s)
python main.py \
  --qaf "eval_data/test_python_rust_CLCCD.jsonl" \
  --custom_cfg "config/qwen3_4b_codestep_rust_v2_1gpu.yaml" \
  --model_dir "models/Qwen3-4B" \
  --branch "codestep-clccd-v2"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"
