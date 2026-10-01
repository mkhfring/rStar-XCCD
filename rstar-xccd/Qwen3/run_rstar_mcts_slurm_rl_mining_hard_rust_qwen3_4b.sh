#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --job-name=q3-4b-rl-mining-hard-rust
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

# Qwen3-4B "hard CLCCD" SFT mining, rust. Source question file is
# eval_data/failed_qwen3_4b_rl_train_rust.jsonl (83/960 rl_train
# questions, built via build_failed_dataset.py from Qwen3-4B's own frozen
# final-for-clccd MCTS result on the FULL test_python_rust_CLCCD_depth_16
# file, restricted to the rl_train split so rl_heldout_python_rust_CLCCD
# stays untouched). Same is_sampling=True mining config as the java hard
# job (this model needs no per-language harness variant -- see the yaml's
# header). 83 questions: pilot-scale, 6h budget is generous margin.
MODEL="models/Qwen3-4B"
QAF="eval_data/failed_qwen3_4b_rl_train_rust.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml"
BRANCH="rl-mining-hard-v1"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
