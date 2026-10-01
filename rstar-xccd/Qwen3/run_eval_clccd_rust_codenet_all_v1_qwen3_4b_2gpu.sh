#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-rust-codenet-all-v1
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

# sft_checkpoints/qwen3-4b-codenet-all-v1 (jobs 3745966/3745967, 2026-09-22):
# fine-tuned on 6462 safe leaves mined from eval_data/codenet_train_python_
# {java,rust}.jsonl (java+rust combined). Track-B-style data, disjoint from
# the CLCCD test files, so eval on the FULL CLCCD rust test set is valid
# (see run_eval_clccd_java_codenet_all_v1_qwen3_4b_2gpu.sh for the full
# rationale). Same eval config as the published Qwen3-4B baseline (job
# 2429954, COMPLETED 05:00:52, F1 0.8739 -- FINAL_FOR_CLCCD.md). Compare
# against run_eval_clccd_rust_codenet_rust_only_v1_qwen3_4b_2gpu.sh (same
# recipe, rust-only 3134-example training data) to see whether the java
# examples mixed into this checkpoint's training helped or hurt rust
# specifically.
#
# TIME BUDGET: base-model rust run took 5h01m for these 1200 questions.
# Budgeted well above the worst-case 8x slowdown seen on the fine-tuned
# checkpoint's earlier heldout runs (same fp32-save inference-speed issue,
# see the java launcher's comment for detail).
MODEL="sft_checkpoints/qwen3-4b-codenet-all-v1"
QAF="eval_data/test_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="assert-consistency-score"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
