#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-eval-clccd-java-base-newrule
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

# NEW-RULE CONFIRMATION RUN (MCTS_SEARCH_SCORING.md, "Scoring review, 2026-09-24").
# Fresh base Qwen3-4B trees on the FULL CLCCD java file, same model/config as
# the published baseline. main.py writes the usual _result (CURRENT rule);
# rescore_runs.py then writes <output>_result_NEWRULE_tested_agg with the old,
# current and NEW ("tested non-clone votes clone") rules, broken down by
# rl_train / rl_heldout / rl_heldout_strict, plus the rl_train-calibrated
# held-out number. The branch tag "newrule-tested-agg" is in the output name.
# Predictions from offline re-scoring of the existing baseline trees:
#   rust rl_heldout 0.804 -> 0.891, rl_heldout_strict 0.800 -> 0.879
#   java rl_heldout 0.908 -> 0.963
MODEL="models/Qwen3-4B"
QAF="eval_data/test_python_java_CLCCD.jsonl"
CFG="config/my_test_mcts_qwen3_4b_depth_16_2gpu_no_sampling.yaml"
BRANCH="newrule-tested-agg"

python main.py \
  --qaf "$QAF" \
  --custom_cfg "$CFG" \
  --model_dir "$MODEL" \
  --branch "$BRANCH"

OUT=$(ls -t eval_data/test_python_java_CLCCD_depth_16.jsonl.mcts.Qwen3-4B.${BRANCH}.${SLURM_JOB_ACCOUNT}.*.jsonl 2>/dev/null | head -1)
if [ -n "$OUT" ] && [ -f "${OUT}_result" ]; then
  python rescore_runs.py --lang java --calibrate "$OUT" --report_out "${OUT}_result_NEWRULE_tested_agg"
else
  echo "main.py did not finish (no ${OUT}_result); resubmit this script to resume, then rescore."
fi
