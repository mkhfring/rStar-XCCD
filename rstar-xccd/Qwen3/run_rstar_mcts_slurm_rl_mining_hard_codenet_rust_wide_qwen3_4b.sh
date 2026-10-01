#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --job-name=q3-4b-rl-mining-hard-codenet-rust-wide
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

REPO="/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd"

RUNDIR="${SLURM_TMPDIR:-/tmp}/rundir_${SLURM_JOB_ID:-$$}"
mkdir -p "$RUNDIR"
for entry in "$REPO"/* "$REPO"/.[!.]*; do
  [ -e "$entry" ] && ln -sfn "$entry" "$RUNDIR/$(basename "$entry")"
done
cd "$RUNDIR"

echo "Current directory: $(pwd) (symlink farm over $REPO)"
echo "Running SLURM job script: $0"
echo "Git branch: $(git -C "$REPO/.." rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO/.." rev-parse --short HEAD)"

module load StdEnv/2023 cudacore/.12.6.2 nccl/2.27.7 java/17.0.6
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# best_of is a sampling param this codebase relies on for candidate
# re-ranking. Force the legacy V0 engine, which still supports it and
# still supports Qwen3.
export VLLM_USE_V1=0

VENV="$REPO/../venv-qwen3/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "FATAL: venv not found at $VENV" >&2
  exit 1
fi
source "$VENV"

# WIDE re-mining of the rust hard-codenet subset. Same 93 questions as the
# standard rl-mining-hard-codenet-v1 rust run (eval_data/
# failed_codenet_python_rust.jsonl), but with n_generate_sample/best_of
# doubled (4->8) and iterations raised 3->12 so the search can actually use
# the wider branching instead of being capped at depth 1 by the MCTS
# depth-guard (see config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_
# rl_mining_wide_rust.yaml's header for the full rationale, including the
# vLLM best_of>=n constraint). The standard run only recovered 11/93
# (11.8%) questions with a safe leaf, from just 11 unique problems (46
# leaves total, 2-6 near-duplicates each) -- SFT_DATA_PROCESS.txt section
# 8. This is a second attempt at the SAME 93 questions to see whether more
# of the 82 still-unrecovered ones are reachable with a wider budget, or
# genuinely out of reach at this model scale.
#
# Distinct branch name (rl-mining-hard-codenet-wide-v1) keeps this
# separate from both the rl_train-derived rl-mining-hard-v1 branch and the
# standard-budget rl-mining-hard-codenet-v1 branch.
MODEL="models/Qwen3-4B"
QAF="eval_data/failed_codenet_python_rust.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining_wide_rust.yaml"
BRANCH="rl-mining-hard-codenet-wide-v1"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
