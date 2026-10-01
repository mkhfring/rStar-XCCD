#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-sft-hard-codenet-java-only-v2
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

# SFT_DATA_PROCESS.txt section 9 (2026-09-24): dataset curated with
# score_rl_rollouts.py strict mode + per-leaf manual review + differential
# testing (diff_test_pairs.py); teacher/repair traces carry only REAL harness
# observations (build_teacher_traces.py). Same small-batch recipe as the
# hard-codenet v1 pilots (effective batch 8, 4 epochs) -- NOT the Track-B
# effective-batch-64 recipe (collapses to ~2 optimizer steps at this size).
# Track B (CodeNet) data: full-CLCCD-evaluable. v2 = strict + reviewed leaves.
MODEL="models/Qwen3-4B"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml"
SFT_DATA="eval_data/sft_hard_codenet_java_only_v2_full_prompt.jsonl"
python train/build_full_prompts.py \
  --input eval_data/sft_hard_codenet_java_only_v2.jsonl \
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
export MASTER_PORT="1944"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

# ABSOLUTE path under $REPO, not relative under $RUNDIR (see
# run_sft_hard_v1_qwen3_4b_4gpu.sh for why: a relative path here would
# silently write the checkpoint to node-local scratch and lose it when the
# job ends).
OUTPUT_DIR="$REPO/sft_checkpoints/qwen3-4b-hard-codenet-java-only-v2"

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
