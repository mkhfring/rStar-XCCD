#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Twin guard (both GPU accounts; see twin_guard.inc)
source /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/Qwen3/twin_guard.inc

# CODE-STEP v2 PILOT (2026-10-03, user: kill v1, fix, resubmit): branch codestep-prompt, worktree rStar-XCCD-codestep.
# 200-pair stratified CLCCD sample (seed 1, DISJOINT from the seed-0 v1 pilot used to find the v1 failures; java 100/100, rust 50/150):
# eval_data/e16a/codestep_pilot/pilot2_python_java_CLCCD.jsonl. Compare against the frozen extension
# (clccd-ext-it8v3 full run, main tree) on the same indices. Resume: resubmit unchanged.
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-codestep/rstar-xccd

echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0
export PATH=$HOME/.cargo/bin:$PATH
source ../venv-qwen3/bin/activate

START=$(date +%s)
python main.py \
  --qaf "eval_data/e16a/codestep_pilot/pilot2_python_java_CLCCD.jsonl" \
  --custom_cfg "config/qwen3_4b_codestep_java_v2_1gpu.yaml" \
  --model_dir "models/Qwen3-4B" \
  --branch "codestep-v2-pilot"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"
