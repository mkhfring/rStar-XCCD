#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-rust-codenet-rust-only-v1-fard
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

# sft_checkpoints/qwen3-4b-codenet-rust-only-v1 (jobs 3745968/3745969,
# 2026-09-22): fine-tuned on 3134 safe leaves mined from
# eval_data/codenet_train_python_rust.jsonl ONLY -- no java in the training
# data, so java eval is skipped (nothing to plausibly transfer, mirroring
# how run_sft_hard_java_v1_qwen3_4b_4gpu.sh's java-only checkpoint skipped
# rust eval). Track-B-style data, disjoint from the CLCCD test files, so
# eval on the FULL CLCCD rust test set is valid (see
# run_eval_clccd_java_codenet_all_v1_qwen3_4b_2gpu.sh for the full
# rationale). Same eval config as the published Qwen3-4B baseline (job
# 2429954, COMPLETED 05:00:52, F1 0.8739 -- FINAL_FOR_CLCCD.md). Compare
# against run_eval_clccd_rust_codenet_all_v1_qwen3_4b_2gpu.sh (same recipe,
# same source mining run, but WITH 3328 java leaves mixed in) to see
# whether isolating rust-only data helps or hurts rust performance versus
# the combined mix.
#
# TIME BUDGET: base-model rust run took 5h01m for these 1200 questions.
# Budgeted well above the worst-case 8x slowdown seen on the fine-tuned
# checkpoint's earlier heldout runs (same fp32-save inference-speed issue,
# see run_eval_clccd_java_codenet_all_v1_qwen3_4b_2gpu.sh's comment for
# detail).
MODEL="sft_checkpoints/qwen3-4b-codenet-rust-only-v1"
QAF="eval_data/test_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="assert-consistency-score"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
