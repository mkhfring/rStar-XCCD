#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-fard_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-sft-codenet-all-v1-fard
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
# cluster; the module ships a prebuilt wheel instead).
module load StdEnv/2023 gcc arrow/15.0.1 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# Must use venv-qwen3, not the base venv: base venv's transformers (4.45.2)
# predates Qwen3 support entirely. venv-qwen3's transformers (4.53.2) has
# it, including Qwen3DecoderLayer for FSDP auto-wrap below.
VENV="$REPO/../venv-qwen3/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "FATAL: venv not found at $VENV" >&2
  exit 1
fi
source "$VENV"

# codenet mining, BOTH languages combined (jobs 3622157 java-fard /
# 3622171 rust-fard, 2026-09-21/22, COMPLETED): unlike the hard-v1 pilot
# (rl_train questions the model had already failed once), this mines
# eval_data/codenet_train_python_{java,rust}.jsonl -- the same Track-B
# CodeNet-derived pool used for the Qwen2.5-7B Track B run
# (Qwen2.5/run_sft_trackB_qwen2.5_7b_4gpu.sh), which per SFT_DATA_PROCESS.txt
# section 2 NEVER touches the CLCCD test files. That means, unlike the
# hard-v1/hard-java-v1 checkpoints (which must only be evaluated on
# rl_heldout_python_{java,rust}_CLCCD.jsonl to avoid leakage), a model
# trained ONLY on this data can be evaluated on the FULL
# eval_data/test_python_{java,rust}_CLCCD.jsonl -- directly comparable to
# FINAL_FOR_CLCCD.md's published Qwen3-4B baseline (java F1 0.9454, rust F1
# 0.8739).
#
# Curated via score_rl_rollouts.py's safe/dangerous execution filter:
#   java: 1720/2000 questions (86.0%) yielded >=1 safe leaf, 3328 safe
#         leaves, 979 dangerous excluded -> eval_data/sft_codenet_qwen3_4b_java.jsonl
#   rust: 1624/2000 questions (81.2%) yielded >=1 safe leaf, 3134 safe
#         leaves, 1282 dangerous excluded -> eval_data/sft_codenet_qwen3_4b_rust.jsonl
# Combined dataset: 6462 examples -- this IS the real Track-B-scale dataset
# for Qwen3-4B (the 59-example hard-v1 pilot was always a pilot, not this).
# Compare against sft_checkpoints/qwen3-4b-codenet-rust-only-v1 (same recipe,
# rust-only 3134 examples) to see whether the java examples in the combined
# mix help or hurt rust performance specifically.
MODEL="models/Qwen3-4B"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml"
SFT_DATA="eval_data/sft_codenet_qwen3_4b_java_rust_full_prompt.jsonl"
python train/build_full_prompts.py \
  --input eval_data/sft_codenet_qwen3_4b_java.jsonl \
  --cfg "$CFG" \
  --input eval_data/sft_codenet_qwen3_4b_rust.jsonl \
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
export MASTER_PORT="1941"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

# ABSOLUTE path under $REPO, not relative under $RUNDIR (see
# run_sft_hard_v1_qwen3_4b_4gpu.sh for why: a relative path here would
# silently write the checkpoint to node-local scratch and lose it when the
# job ends). Name encodes exactly what this was trained on: qwen3-4b base
# model, codenet-mined data, ALL languages (java+rust) combined, v1.
OUTPUT_DIR="$REPO/sft_checkpoints/qwen3-4b-codenet-all-v1"

# per_device_train_batch_size=2 is the value already validated not to OOM
# for Qwen3-4B with sdpa+gradient_checkpointing (hard-v1 pilot). Bumped
# gradient_accumulation_steps 1->8 (effective batch 64) and epochs 4->2
# now that the dataset is real-sized (6462 examples, ~101 steps/epoch, ~202
# steps total) rather than the 59-example pilot that needed a tiny
# effective batch just to get any optimizer steps at all. This scaling is a
# judgment call interpolating between the hard-v1 pilot (effective batch 8,
# 59 examples) and the Qwen2.5-7B Track B recipe (effective batch 128,
# 21853 examples), not an independently tuned recipe for this exact size.
python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=4 --use_env train/train_SFT.py \
    --model_name_or_path "$MODEL" \
    --model_max_length 8192 \
    --data_path "$SFT_DATA" \
    --data_length 10000000 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
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
