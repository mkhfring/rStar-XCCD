#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --output=Qwen3/%x_%j.log.out
#SBATCH --error=Qwen3/%x_%j.log.err
# Twin guard (both GPU accounts; see twin_guard.inc)
source /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/Qwen3/twin_guard.inc
# Baseline revision (BASELINE_REVISION_PLAN_2026-09-28.txt): literature prompts
# (CLCCD six single-call prompts) on one GPU. All arguments are passed through to
# pure_inference/run_clccd_paper_prompts.py, e.g.
#   sbatch --job-name=lit-q3-8b-think-test-java Qwen3/run_litprompts_1gpu.sh \
#     --model_dir Qwen3-8B --datasets java --split test --enable_thinking \
#     --max_tokens 8192 --max_model_len 32768 --temperature 0.6 --top_p 0.95 --top_k 20 --seed 1
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0
source ../venv-qwen3/bin/activate
echo "ARGS: $*"
START=$(date +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python pure_inference/run_clccd_paper_prompts.py --tp 1 "$@"
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))"
