#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --job-name=q3-8b-retry-java-control
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

# CONTROL RUN for the code2-exec-lang-fewshot experiment (see RESUME_HERE.txt
# section 7c). Same 45 python-java samples that both Qwen2.5-Coder-7B and
# Qwen3-8B got wrong in the final-for-clccd runs, same LD_PRELOAD-fixed
# rustc/javac path, but under the GENERIC code2-exec config
# (few_shot_imprvoed.json + mcts_prompt_code2_exec.json) instead of the
# language-specific fewshot/v2 prompts -- i.e. the model is told it CAN
# compile/run Code 2, with no worked java-specific examples and without the
# rule-7 rewrite that blocked the "not code clones" early exit.
#
# Purpose: v1 was 10/45 correct, v2 forced uptake to 33/45 attempts but fell
# to 9/45 correct -- both at temperature=0.7 with no control. This run
# isolates rerun variance (plus the bare code2-exec capability, uptake
# unforced) so those numbers can be judged against a same-conditions
# baseline instead of the original final-for-clccd 0/45.
MODEL="models/Qwen3-8B"
QAF="eval_data/failed_both_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_8b_depth_16_1gpu_no_sampling_code2_exec.yaml"
BRANCH="code2-exec-generic-control"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
