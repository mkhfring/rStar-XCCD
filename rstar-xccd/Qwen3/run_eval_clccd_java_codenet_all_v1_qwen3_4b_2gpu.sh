#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-java-codenet-all-v1
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
# {java,rust}.jsonl (java+rust combined). This is Track-B-style data (per
# SFT_DATA_PROCESS.txt section 2 / Qwen2.5/run_sft_trackB_qwen2.5_7b_4gpu.sh
# precedent) drawn from a CodeNet pool disjoint from the CLCCD test files --
# unlike the rl_train-derived hard-v1/hard-java-v1 pilots, this checkpoint
# can be evaluated on the FULL CLCCD test set directly, not just a held-out
# subset. Same eval config as the published Qwen3-4B baseline (job 2429953,
# COMPLETED 08:13:05, F1 0.9454 -- see FINAL_FOR_CLCCD.md), so this number
# is directly comparable to that baseline and to
# run_eval_clccd_rust_codenet_all_v1_qwen3_4b_2gpu.sh /
# run_eval_clccd_rust_codenet_rust_only_v1_qwen3_4b_2gpu.sh.
#
# TIME BUDGET: the base-model java run took 8h13m for these same 1800
# questions. The fine-tuned checkpoint's earlier heldout runs (hard-v1,
# 2026-09-22) showed 3-8x slower per-question inference than the base
# model, plausibly because train_SFT.py's HF save ends up fp32 (~17.6GB for
# this 4B model) rather than bf16, so vLLM infers in fp32. Budgeted well
# above the worst-case extrapolation (8x * 8h13m =~ 66h) to avoid a repeat
# timeout; if it finishes much faster, no harm done.
MODEL="sft_checkpoints/qwen3-4b-codenet-all-v1"
QAF="eval_data/test_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="assert-consistency-score"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
