#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-heldout-java-baseline-fard
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Same repo/cd pattern as the assert-consistency-score baseline scripts
# (e.g. run_rstar_mcts_slurm_clccd_pj_fard_qwen3_4b_depths_16_2gpu_assert_consistency_score.sh)
# that produced the FINAL_FOR_CLCCD.md numbers -- cd directly into the repo,
# no RUNDIR symlink farm, so there's no risk of the SFT launchers' lost-
# checkpoint bug (output here is just an eval_data/*.jsonl result file,
# which lands directly in the real repo either way).
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

# vLLM V1 (the default in 0.8.5.post1) dropped support for the `best_of`
# sampling param this codebase relies on for candidate re-ranking. Force
# the legacy V0 engine, which still supports it and still supports Qwen3.
export VLLM_USE_V1=0

source ../venv-qwen3/bin/activate

# BASELINE (un-fine-tuned) Qwen3-4B on the java rl_heldout split ONLY
# (eval_data/rl_heldout_python_java_CLCCD.jsonl, 360 questions) -- NOT the
# full test_python_java_CLCCD.jsonl (1800 q) FINAL_FOR_CLCCD.md's 0.9454
# number comes from. That number is not usable as this experiment's
# baseline: rl_heldout is a specific 360-question subset of it, and could
# be an easier or harder slice than the full-file average. This run exists
# purely so sft_hard_v1 / sft_hard_java_v1's heldout numbers below have a
# same-subset baseline to compare against -- same config (depth 16,
# n_generate_sample=2, is_sampling=False, 2 rollouts,
# assert-consistency-score scoring) as the FINAL_FOR_CLCCD.md runs, only
# the question file and --branch differ.
MODEL="models/Qwen3-4B"
QAF="eval_data/rl_heldout_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="heldout-baseline"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH" \
  --no-resume
