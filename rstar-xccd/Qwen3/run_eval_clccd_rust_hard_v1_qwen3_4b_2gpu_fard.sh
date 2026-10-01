#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-rust-hard-v1-TAINTED-fard
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Twin guard: the same job is also queued under def-khajezad_gpu; whichever starts first wins.
TWIN_ACCOUNT="def-khajezad_gpu"
TWIN_NAME="${SLURM_JOB_NAME%-fard}"
TWIN_RUNNING=$(squeue -h -u "$USER" -A "$TWIN_ACCOUNT" -n "$TWIN_NAME" -t RUNNING -o %i)
if [ -n "$TWIN_RUNNING" ]; then
  echo "Twin job $TWIN_RUNNING ($TWIN_ACCOUNT) is already running; exiting without work."
  exit 0
fi
for t in $(squeue -h -u "$USER" -A "$TWIN_ACCOUNT" -n "$TWIN_NAME" -t PENDING -o %i); do
  echo "Started first: cancelling pending twin $t ($TWIN_ACCOUNT)"
  scancel "$t"
done

cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

# *** RESULT IS LEAKAGE-CONTAMINATED -- DO NOT REPORT AS A GENERALIZATION
# NUMBER WITHOUT THIS CAVEAT. ***
#
# sft_checkpoints/qwen3-4b-hard-v1 (jobs 3645950/3645952, 2026-09-22) was
# fine-tuned on rejection-sampled traces mined from
# eval_data/failed_qwen3_4b_rl_train_{java,rust}.jsonl -- questions drawn
# from eval_data/rl_train_python_{java,rust}_CLCCD.jsonl (960 rust
# questions), specifically the 12 rust questions that yielded the 19 safe
# leaves in eval_data/sft_hard_v1_rust.jsonl. rl_train_python_rust_CLCCD is
# a strict subset of THIS job's QAF (eval_data/test_python_rust_CLCCD.jsonl,
# the full 1200-question file: verified rl_train (960) + rl_heldout (240) =
# full (1200), zero index overlap between the two splits). That means this
# run's 1200 questions include the exact 12 the model was fine-tuned on --
# real train/eval leakage, not a hypothetical concern.
#
# Every other hard-v1/hard-java-v1 launcher in this project (e.g.
# run_eval_heldout_rust_hard_v1_qwen3_4b_2gpu.sh) deliberately evaluates
# ONLY on rl_heldout_python_{java,rust}_CLCCD.jsonl to avoid exactly this;
# that clean run already completed (job 3732069, F1 0.9217 on the 240-q
# heldout set). THIS run was requested anyway, on the full leakage-tainted
# file, as an explicit, acknowledged exception -- read its F1 as an upper
# bound / not-directly-comparable number, not a valid generalization
# estimate. BRANCH is tagged accordingly so the output filename carries
# this warning forward.
#
# TIME BUDGET: mirrors run_eval_clccd_rust_codenet_all_v1_qwen3_4b_2gpu.sh
# (same checkpoint size class, ~80G/fp32-save, same 1200-question file);
# base-model rust run took 5h01m, fine-tuned checkpoints have shown up to
# 8x slower inference in earlier runs.
MODEL="sft_checkpoints/qwen3-4b-hard-v1"
QAF="eval_data/test_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="clccd-full-LEAKAGE-TAINTED"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
