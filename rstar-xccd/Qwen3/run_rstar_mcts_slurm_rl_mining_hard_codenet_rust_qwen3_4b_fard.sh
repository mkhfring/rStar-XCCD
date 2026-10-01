#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --job-name=q3-4b-rl-mining-hard-codenet-rust-fard
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

# Qwen3-4B "hard CodeNet" SFT mining, rust. Source question file is
# eval_data/failed_codenet_python_rust.jsonl (93/2000 CodeNet-mining
# questions, built via build_failed_dataset.py from the codenet-mining-v1
# tree's own aggregate clone/non-clone verdict --
# eval_data/codenet_train_python_rust_depth_16.jsonl.mcts.Qwen3-4B.
# rl-mining-codenet-v1.def-fard_gpu.20260921203724.jsonl -- i.e. questions
# Qwen3-4B got wrong across its own sampled rollouts, NOT rl_train-derived
# like rl-mining-hard-v1's failed_qwen3_4b_rl_train_rust.jsonl). Checked:
# only 1/93 of these questions already have a safe leaf in that mining
# output, so this re-mines them fresh with a new is_sampling=True pass to
# see if additional rollouts land on a correct, trustworthy leaf. Distinct
# branch name (rl-mining-hard-codenet-v1) keeps this separate from the
# rl_train-derived rl-mining-hard-v1 branch. Same CFG as every other Qwen3-4B
# mining job -- no per-language variant needed for this model.
#
# Pair with the java job's output at the SFT-dataset stage to also build a
# java+rust-combined hard-codenet dataset, mirroring how codenet-all-v1 was
# built from separately-mined codenet_train_python_{java,rust} files -- no
# third "combined" mining run is needed.
MODEL="models/Qwen3-4B"
QAF="eval_data/failed_codenet_python_rust.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml"
BRANCH="rl-mining-hard-codenet-v1"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
