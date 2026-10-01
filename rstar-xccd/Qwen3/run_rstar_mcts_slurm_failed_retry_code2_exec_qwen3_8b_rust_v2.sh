#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --job-name=q3-8b-retry-rust-v2
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# See the final-for-clccd 8B script for why this path is absolute rather than
# derived from ${BASH_SOURCE[0]} or $SLURM_SUBMIT_DIR.
REPO="/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd"

# Run from a private working directory -- a symlink farm over the repo --
# rather than from the repo itself. Every step writes scratch code into the
# working directory (candidate_code.py, <Class>.java and the .class files
# javac emits, candidate_code2.rs and its binary, plus whatever the model
# names its own test files), and cleanup_generated_code() deletes those at
# the end of each step. Two jobs sharing one working directory therefore
# race: one can delete a file the other is about to run. Reads and the
# result .jsonl still go to the real repo through the symlinks, so output
# lands in eval_data/ exactly as before; only the scratch stays node-local,
# which also keeps the repo root clean.
RUNDIR="${SLURM_TMPDIR:-/tmp}/rundir_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNDIR"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
  [ -e "$entry" ] && ln -sfn "$entry" "$RUNDIR/$(basename "$entry")"
done
cd "$RUNDIR"

echo "Current directory: $(pwd) (symlink farm over $REPO)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C "$REPO/.." rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO/.." rev-parse --short HEAD)"

# Preload a newer NCCL than the venv's PyTorch links against; see the
# final-for-clccd 8B script. Modules must load BEFORE the venv activates.
# java is new here: javac/java need to be on PATH for the model's Code 2
# (Java) subprocess tests to actually succeed instead of failing with
# FileNotFoundError -- see stage_code2() in
# rstar_deepthink/tools/python_tool.py. Rust needs no module: rustc/cargo
# are already on PATH via ~/.cargo/bin.
module load StdEnv/2023 gcc arrow/15.0.1 python/3.11 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# vLLM V1 (the default in 0.8.5.post1) dropped support for the `best_of`
# sampling param this codebase relies on for candidate re-ranking. Force
# the legacy V0 engine, which still supports it and still supports Qwen3.
export VLLM_USE_V1=0

# venv-qwen3, NOT venv: see the final-for-clccd 8B script.
# Anchored to $REPO, not CWD: the working directory is now RUNDIR,
# so a relative ../venv-qwen3 resolves under /localscratch and misses.
VENV="$REPO/../venv-qwen3/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "FATAL: venv not found at $VENV" >&2
  exit 1
fi
source "$VENV"

# Retry of the samples BOTH Qwen2.5-Coder-7B and Qwen3-8B got wrong in the
# final-for-clccd runs (see build_failed_dataset.py; 20 python-rust samples,
# 19 of them false negatives on true clones). Baseline is 0/20 correct by
# construction, so any correct answer here is a recovered sample.
#
# Note the python-rust set is drawn from the 723-sample prefix the Qwen3-8B
# rust run actually reached before its wall clock killed it -- the remaining
# 477 questions were never scored by that model, so more joint failures
# likely exist beyond this set.
#
# What is under test is the language-specific prompt pair:
# few_shot_code2_exec_python_rust.json, whose worked examples run BOTH sides
# on the same stdin and compare actual output, and
# mcts_prompt_code2_exec_python_rust.json, whose rule 7 no longer lets the
# model exit to "not code clones" on a merely suspected behavioural
# difference -- the exit that blocked Code-2 execution on 12/20 (7B) and
# 17/20 (8B) of exactly these samples.
#
# temperature is 0.7, so a recovered sample is prompt effect plus rerun
# variance; a control run under the generic _code2_exec config separates
# the two.
MODEL="models/Qwen3-8B"
QAF="eval_data/failed_both_python_rust_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_8b_depth_16_1gpu_no_sampling_code2_exec_rust_v2.yaml"
BRANCH="code2-exec-lang-fewshot-v2"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
