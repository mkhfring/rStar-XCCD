#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --account=def-khajezad_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --job-name=q3-4b-sft-hard-java-v1
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
# cluster; the module ships a prebuilt wheel instead). Same issue
# Qwen2.5-7B's SFT launcher hit (job 3081888); see SFT_DATA_PROCESS.txt
# section 7.
module load StdEnv/2023 gcc arrow/15.0.1 cudacore/.12.6.2 nccl/2.27.7
export LD_PRELOAD="$EBROOTNCCL/lib/libnccl.so.2"

# Must use venv-qwen3, not the base venv: base venv's transformers
# (4.45.2) predates Qwen3 support entirely (confirmed directly:
# ModuleNotFoundError: transformers.models.qwen3). venv-qwen3's
# transformers (4.53.2) has it, including Qwen3DecoderLayer for FSDP
# auto-wrap below.
VENV="$REPO/../venv-qwen3/bin/activate"
if [ ! -f "$VENV" ]; then
  echo "FATAL: venv not found at $VENV" >&2
  exit 1
fi
source "$VENV"

# python-java ONLY, rl-mining "hard-v1" track (job 3609245, 2026-09-21):
# the subset of rl_train_python_java_CLCCD Qwen3-4B's own frozen
# final-for-clccd config previously got WRONG, re-mined with wide
# sampling (n_generate_sample=best_of=4, iterations=3, temp=0.9) to see
# if the model can rediscover a correct, non-fabricated trace on its own
# hard cases. Curated via score_rl_rollouts.py's safe/dangerous filter
# (SFT_DATA_PROCESS.txt section 4): 22/108 questions (20.4%) yielded >=1
# safe leaf, 40 safe leaves, 9 dangerous excluded ->
# eval_data/sft_hard_v1_java.jsonl. Rust deliberately excluded here (see
# run_sft_hard_v1_qwen3_4b_4gpu.sh / job 3621719 for the combined
# java+rust run) -- this run is java-only by request.
#
# Only rl_train_python_java_CLCCD.jsonl was ever touched -- evaluate
# whatever this produces on rl_heldout_python_java_CLCCD ONLY, per
# SFT_DATA_PROCESS.txt section 2's rule, never the full CLCCD test file
# or rl_train itself.
#
# DATASET IS TINY: 40 examples. The Qwen2.5-7B Track B recipe this
# launcher is otherwise modeled on (run_sft_trackB_qwen2.5_7b_4gpu.sh)
# used per_device_train_batch_size=4 x 4 GPUs x grad_accum=8 = effective
# batch 128, tuned for a 21853-example dataset. Copied as-is here, an
# optimizer step's 128-example accumulation window would almost never
# fill across just 2 epochs (80 sample-views total) -- training would
# silently do next to nothing. Scaled down instead: batch=2 x 4 GPUs x
# grad_accum=1 = effective batch 8 (~5 optimizer steps/epoch on 40
# examples), and epochs raised 2->4 since a tiny dataset needs more
# passes to have any effect. This is a judgment call, not an established
# recipe from this project -- treat the resulting checkpoint as a first
# pilot data point, not a tuned result.
MODEL="models/Qwen3-4B"
CFG="config/my_test_mcts_qwen3_4b_depth_16_1gpu_sampling_rl_mining.yaml"
SFT_DATA="eval_data/sft_hard_v1_java_full_prompt.jsonl"
python train/build_full_prompts.py \
  --input eval_data/sft_hard_v1_java.jsonl \
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
export MASTER_PORT="1940"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"

# ABSOLUTE path under $REPO, not relative under $RUNDIR: sft_checkpoints/
# did not exist in $REPO when the symlink farm above was built, so it was
# never symlinked back -- a relative OUTPUT_DIR here would write the
# trained model to node-local $SLURM_TMPDIR scratch, silently lost when
# the job ends. Confirmed the hard way: jobs 3632724/3632727 both
# COMPLETED with real decreasing loss and no checkpoint ever reached
# $REPO/sft_checkpoints.
OUTPUT_DIR="$REPO/sft_checkpoints/qwen3-4b-hard-java-v1"

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
