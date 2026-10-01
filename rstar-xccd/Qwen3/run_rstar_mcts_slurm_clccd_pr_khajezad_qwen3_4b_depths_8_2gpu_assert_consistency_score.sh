#!/bin/bash

#SBATCH --time=10:00:00

#SBATCH --ntasks=1

#SBATCH --account=def-khajezad_gpu

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:a100:2

#SBATCH --mem=256G

#SBATCH --job-name=q3-clccd-pr-4b-mcts-d8-assert-consistency-score-khajezad

#SBATCH --output=%x_%j.log.out

#SBATCH --error=%x_%j.log.err


# Go to the project root directly. NOT ${BASH_SOURCE[0]} or
# $SLURM_SUBMIT_DIR/..: SLURM copies the script into a spool dir on
# the compute node before running it (so BASH_SOURCE would resolve to
# the spool path), and SLURM_SUBMIT_DIR is wherever `sbatch` was
# invoked from, not this script's location, so a relative ".." is
# fragile to how the job is submitted.
#
# This is the assert-consistency-score branch checkout (a separate git
# clone from the production /project/6104180/khajezad/rStar-XCCD/rstar-xccd
# used by other scripts), so results land in their own eval_data/ and are
# tagged with --branch below, keeping them fully separate from other runs.
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

# Run rStar MCTS, max_depth=8, on the CLCCD-derived Python/Rust clone-detection
# test set (300 clone + 900 non-clone, built via the CLCCD Java-problem-id
# bridge; see CLCCD_DATASET_CONSTRUCTION.md), not the pre-existing
# hand-curated test_same_python_rust.jsonl file. Time budget capped at 10h.
#
# Uses the assert-consistency-score branch (on top of search-score-update):
# see MCTS_SEARCH_SCORING.md. create_child() in mcts.py defers scoring a
# completed assertion (pass or AssertionError) until eval_final_answer()
# resolves, then scores it by whether it agrees with the branch's own
# eventual conclusion -- never against ground truth.
#
# Uses config/my_test_mcts_qwen3_4b_depth_8_2gpu_no_sampling.yaml, a copy of
# the shared my_test_mcts_qwen3_4b_depth_8_2gpu.yaml with is_sampling flipped
# to False, so the search never sees the ground-truth label. --branch tags
# the output filename so it can be told apart from runs made with other
# branches/code versions.
#
# This is the same qaf/config/model/branch combo as the def-fard_gpu
# submission of this same script (see the non-"_khajezad" version) -- run
# under a second account in parallel to get GPU time from whichever queue
# clears first. --no-resume is REQUIRED here: main.py's default --resume
# globs for the latest existing output file matching this exact
# qaf/mode/model/branch combo, with no awareness of which account produced
# it. Without --no-resume, this job would find (and start appending into)
# the fard-account job's output file the moment that file exists --
# corrupting it if both jobs are writing concurrently. --no-resume forces
# this job to always start its own independently-timestamped file instead.
MODEL="models/Qwen3-4B"
QAF="eval_data/test_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_8_2gpu_no_sampling.yaml"
BRANCH="assert-consistency-score"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
