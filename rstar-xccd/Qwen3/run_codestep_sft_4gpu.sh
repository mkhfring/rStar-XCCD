#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-codestep-sft
#SBATCH --output=%x_%j.log.out
#SBATCH --error=%x_%j.log.err

# Code-step SFT (training plan TRAINING_PLAN_CODESTEP_2026-10-05.txt, Phase 0.5 / 1).
# Settings come from the environment (sbatch --export=ALL,NAME=...,DATA=...):
#   NAME       checkpoint name -> sft_checkpoints/$NAME (required; new name per run)
#   DATA       jsonl from train/codestep_sft_data.py (required)
#   LR         learning rate (default 1e-5; 0 for the identity smoke test)
#   EPOCHS     default 2
#   MAX_STEPS  optional, overrides EPOCHS (e.g. 2 for the identity test)
#   PROVENANCE default "self,claude_written,claude_edited"
# Submit as a twin-guard pair (both accounts, job names NAME / NAME-fard).
# After the run: python train/check_sft_checkpoint.py --base models/Qwen3-4B --ckpt sft_checkpoints/$NAME [--identical]

source /project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD/rstar-xccd/Qwen3/twin_guard.inc

REPO="/project/6104180/khajezad/rstar-xccd-remote/rStar-XCCD-train/rstar-xccd"
cd "$REPO"
: "${NAME:?set NAME}" "${DATA:?set DATA}"
LR="${LR:-1e-5}"; EPOCHS="${EPOCHS:-2}"; PROVENANCE="${PROVENANCE:-self,claude_written,claude_edited}"
STEPS_ARG=(); [ -n "$MAX_STEPS" ] && STEPS_ARG=(--max_steps "$MAX_STEPS")
echo "Git branch: $(git -C .. rev-parse --abbrev-ref HEAD) @ $(git -C .. rev-parse --short HEAD)"
echo "NAME=$NAME DATA=$DATA LR=$LR EPOCHS=$EPOCHS MAX_STEPS=$MAX_STEPS PROVENANCE=$PROVENANCE"

module load StdEnv/2023 gcc arrow/15.0.1 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"
source ../venv-qwen3/bin/activate

export NCCL_IB_DISABLE=1
export MASTER_ADDR="localhost"
export MASTER_PORT="$((19000 + SLURM_JOB_ID % 1000))"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export FSDP_STATE_DICT_TYPE=FULL_STATE_DICT

OUTPUT_DIR="$REPO/sft_checkpoints/$NAME"
if [ -e "$OUTPUT_DIR/config.json" ]; then
  echo "FATAL: $OUTPUT_DIR already holds a checkpoint; use a new NAME" >&2
  exit 1
fi

START=$(date +%s)
python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    --nproc_per_node=4 --use_env train/train_codestep_sft.py \
    --model_name_or_path models/Qwen3-4B \
    --data_path "$DATA" \
    --provenance "$PROVENANCE" \
    --model_max_length 8192 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" "${STEPS_ARG[@]}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate "$LR" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none \
    --seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap Qwen3DecoderLayer \
    --gradient_checkpointing True \
    --attn_impl sdpa
RC=$?
echo "WALLCLOCK_SECONDS: $(( $(date +%s) - START ))  exit $RC"
[ $RC -eq 0 ] && python train/check_sft_checkpoint.py --base models/Qwen3-4B --ckpt "$OUTPUT_DIR" $([ "$LR" = "0" ] && echo --identical)
