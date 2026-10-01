#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-clccd-pr-4b-mcts-d16-exec-outcome-scoring-khajezad
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Go to the project root directly. NOT ${BASH_SOURCE[0]} or
# $SLURM_SUBMIT_DIR/..: SLURM copies the script into a spool dir on
# the compute node before running it (so BASH_SOURCE would resolve to
# the spool path), and SLURM_SUBMIT_DIR is wherever `sbatch` was
# invoked from, not this script's location, so a relative ".." is
# fragile to how the job is submitted.
#
# This is the exec-outcome-scoring branch checkout (a separate git clone
# from the production /project/6104180/khajezad/rStar-XCCD/rstar-xccd used
# by other scripts), so results land in their own eval_data/ and are tagged
# with --branch below, keeping them fully separate from other runs.
cd /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd

echo "Current directory: $(pwd)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"

# The venv's PyTorch links against nccl/2.18.3 (CUDA 12.2 build) via RPATH,
# which hits "named symbol not found" against this node's newer driver during
# multi-GPU NCCL init. Preload a newer NCCL build to override it (RPATH would
# otherwise take precedence over LD_LIBRARY_PATH, so LD_PRELOAD is required).
# Must load modules BEFORE activating the venv: `module load` rebuilds PATH
# and would otherwise demote the venv's python behind the system one.
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# vLLM V1 (the default in 0.8.5.post1) dropped support for the `best_of`
# sampling param this codebase relies on for candidate re-ranking. Force
# the legacy V0 engine, which still supports it and still supports Qwen3.
export VLLM_USE_V1=0

# Activate environment
source ../venv-qwen3/bin/activate

# Run rStar MCTS, max_depth=16, on the CLCCD-derived Python/Rust
# clone-detection test set (see CLCCD_DATASET_CONSTRUCTION.md).
#
# Uses the exec-outcome-scoring branch, which replaces the search's
# execution scoring after a node-level audit of the assert-consistency-score
# rollouts found it was rewarding the wrong things -- see
# MCTS_SEARCH_SCORING.md for the measurements and the five changes.
#
# Config is the depth-16 n=2 no-sampling config (is_sampling: False, so the
# search never sees the ground-truth label; that is also the code-level
# default in rstar_deepthink/config.py). --branch tags the output filename
# with the branch so these results can be told apart from -- and never
# accidentally resumed from -- runs made with other code versions.
# --no-resume forces a fresh output file rather than appending to any
# existing matching run.
MODEL="models/Qwen3-4B"
QAF="eval_data/test_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="exec-outcome-scoring"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
