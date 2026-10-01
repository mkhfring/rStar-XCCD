#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-sft-hard-v1-fard
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

# arrow provides pyarrow, which transformers' `datasets` import needs --
# must be loaded BEFORE activating the venv (pip install fails on this
# cluster; the module ships a prebuilt wheel instead). Same issue
# Qwen2.5-7B's SFT launcher hit (job 3081888); see SFT_DATA_PROCESS.txt
# section 7.
module load StdEnv/2023 gcc arrow/15.0.1 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# UNLIKE the Qwen2.5-7B SFT launcher: must use venv-qwen3, not the base
# venv. transformers in the base venv (4.45.2) predates Qwen3 support
# entirely (KeyError: 'qwen3' / ModuleNotFoundError:
# transformers.models.qwen3 -- confirmed directly). venv-qwen3's
# transformers (4.53.2) has it, including Qwen3DecoderLayer for FSDP
# auto-wrap below.
VENV="$REPO/../venv-qwen3/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "FATAL: venv not found at $VENV" >&2
  exit 1
fi
source "$VENV"

# rl-mining "hard-v1" track only (jobs 3609245 java / 3609249 rust,
# 2026-09-21): the subset of rl_train Qwen3-4B's own frozen
# final-for-clccd config previously got WRONG, re-mined with wide
# sampling (n_generate_sample=best_of=4, iterations=3, temp=0.9) to see
# if the model can rediscover a correct, non-fabricated trace on its own
# hard cases. Curated via score_rl_rollouts.py's safe/dangerous filter
# (SFT_DATA_PROCESS.txt section 4):
#   java: 22/108 questions (20.4%) yielded >=1 safe leaf, 40 safe leaves,
#         9 dangerous excluded -> eval_data/sft_hard_v1_java.jsonl
#   rust: 12/83  questions (14.5%) yielded >=1 safe leaf, 19 safe leaves,
#         8 dangerous excluded -> eval_data/sft_hard_v1_rust.jsonl
# Yield is much lower than Track A/B's ~90%+ (SFT_DATA_PROCESS.txt
# section 5) because this set is deliberately adversarial -- every
# question here is one the model already failed once; expected, not a
# bug. Combined dataset is small (59 examples) -- this is a pilot-scale
# check of whether hard-mining works at all, not a claim that 59
# examples is enough to move the needle; RESUME_HERE.txt / next steps
# should decide whether to scale this up (more iterations, more
# candidates, or feeding this mining process back in a second round)
# before reading much into the eval numbers.
#
# Same config mines BOTH languages (see the yaml's own header: unlike
# the Qwen2.5-7B mining configs, Qwen3-4B's best-measured CLCCD result
# didn't need a per-language code2-exec fix, so one config covers both
# --input files below with the same --cfg).
#
# Only rl_train_* was ever touched (both by the original mining and by
# this dataset) -- rl_heldout_python_{java,rust}_CLCCD stays untouched,
# so evaluate whatever this produces on rl_heldout_*, per
# SFT_DATA_PROCESS.txt section 2's rule, NOT the full CLCCD test files
# (unlike a Track-B-only run, which could use the full files).
# DATASET IS TINY: 59 examples. batch=2 x 4 GPUs x grad_accum=1 (effective
# batch 8, ~8 steps/epoch, 4 epochs) -- NOT the Qwen2.5-7B recipe's
# batch=4 x 4 GPUs x grad_accum=8 (effective batch 128), which was copied
# here in an earlier revision and confirmed (job 3632724's log) to yield
# only 2 real optimizer steps total across 2 epochs on this dataset --
# barely any training happened despite a clean COMPLETED exit and a
# real-looking loss log. See run_sft_hard_java_v1_qwen3_4b_4gpu.sh for the
# same fix applied there first.
MODEL="models/Qwen3-4B"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml"
SFT_DATA="eval_data/sft_hard_v1_java_rust_full_prompt.jsonl"
python train/build_full_prompts.py \
  --input eval_data/sft_hard_v1_java.jsonl \
  --cfg "$CFG" \
  --input eval_data/sft_hard_v1_rust.jsonl \
  --cfg "$CFG" \
  --model_dir "$MODEL" \
  --output_file "$SFT_DATA"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NLLC_P2P_LEVEL=NVL
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_BLOCKING_WAIT=0
export FLASH_ATTENTION_DETERMINISTIC=1
export MASTER_ADDR="localhost"
export MASTER_PORT="1939"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

# ABSOLUTE path under $REPO, not relative under $RUNDIR: sft_checkpoints/
# did not exist in $REPO when the symlink farm above was built, so it was
# never symlinked back -- a relative OUTPUT_DIR here would write the
# trained model to node-local $SLURM_TMPDIR scratch, silently lost when
# the job ends. Confirmed the hard way: jobs 3632724/3632727 both
# COMPLETED with real decreasing loss and no checkpoint ever reached
# $REPO/sft_checkpoints.
OUTPUT_DIR="$REPO/sft_checkpoints/qwen3-4b-hard-v1"

# fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer, NOT Qwen2DecoderLayer
# (the Qwen2.5-7B launcher's value) -- Qwen3 has its own decoder layer class
# in transformers; wrapping with the wrong class silently fails to shard
# per-layer and defeats the point of FSDP.
python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=4 --use_env train/train_SFT.py \
    --model_name_or_path "$MODEL" \
    --model_max_length 8192 \
    --data_path "$SFT_DATA" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 100000 \
    --save_total_limit 2 \
    --learning_rate 7e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --gradient_checkpointing True \
    --attn_impl sdpa \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen3DecoderLayer'
