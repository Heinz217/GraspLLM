set -euo pipefail

BACKBONE=""; SOURCE=""; GPUS="0"
BS=8; LR=5e-4; EPOCHS=1; WD=0.0; WARMUP=0.03
MAX_LEN=4096; PROJECTOR_TYPE="vicuna_2layermh"
BASE_MODEL=""; OUT_DIR=""; DEEPSPEED_CFG=""; CONV_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backbone)    BACKBONE="$2";    shift 2;;
        --source)      SOURCE="$2";      shift 2;;
        --gpus)        GPUS="$2";        shift 2;;
        --batch-size)  BS="$2";          shift 2;;
        --lr)          LR="$2";          shift 2;;
        --epochs)      EPOCHS="$2";      shift 2;;
        --weight-decay) WD="$2";         shift 2;;
        --warmup)      WARMUP="$2";      shift 2;;
        --max-len)     MAX_LEN="$2";     shift 2;;
        --projector)   PROJECTOR_TYPE="$2"; shift 2;;
        --base-model)  BASE_MODEL="$2";  shift 2;;
        --out-dir)     OUT_DIR="$2";     shift 2;;
        --deepspeed)   DEEPSPEED_CFG="$2"; shift 2;;
        --conv)        CONV_OVERRIDE="$2"; shift 2;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0;;
        *)  echo "unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -n "$BACKBONE" ]] || { echo "--backbone required (vicuna|mistral|llama3|qwen3|qwen3-moe)"; exit 1; }
[[ -n "$SOURCE"   ]] || { echo "--source required (e.g. arxiv|computer|reddit)"; exit 1; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_ALGO=${NCCL_ALGO:-Ring}

case "$BACKBONE" in
    vicuna)    DEFAULT_MODEL="vicuna";   CONV="v1" ;;
    mistral)   DEFAULT_MODEL="mistral";  CONV="llama_2" ;;
    llama3)    DEFAULT_MODEL="llama3";   CONV="v1" ;;
    qwen3)     DEFAULT_MODEL="qwen3";    CONV="v1" ;;
    qwen3-moe) DEFAULT_MODEL="qwen3-moe";CONV="v1" ;;
    *) echo "unknown backbone: $BACKBONE"; exit 1;;
esac
[[ -n "$CONV_OVERRIDE" ]] && CONV="$CONV_OVERRIDE"

if [[ -z "$BASE_MODEL" ]]; then
    BASE_MODEL=$(python -c "from utils.paths import model_dir; print(model_dir('$DEFAULT_MODEL'))")
fi

CHECKPOINT_ROOT=${GRASPLLM_CHECKPOINT_ROOT:-$REPO/checkpoints}
RUN_NAME=${RUN_NAME:-"grasp-${BACKBONE}-qwen3emb-${PROJECTOR_TYPE}-${SOURCE}"}
[[ -z "$OUT_DIR" ]] && OUT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"
mkdir -p "$OUT_DIR"

NPROC=$(awk -F',' '{print NF}' <<<"$GPUS")
if [[ "$NPROC" -eq 1 ]]; then
    LAUNCH=(python -u train/train_mem.py)
else
    LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC"
            --master_port=$((29500 + RANDOM % 1000))
            train/train_mem.py)
fi

EXTRA_ARGS=()
[[ -n "$DEEPSPEED_CFG" ]] && EXTRA_ARGS+=(--deepspeed "$DEEPSPEED_CFG")

echo "================================================================="
echo " Stage-3 train"
echo "   backbone : $BACKBONE  base=$BASE_MODEL  conv=$CONV"
echo "   source   : $SOURCE    projector=$PROJECTOR_TYPE"
echo "   gpus     : $GPUS  (nproc=$NPROC)  bs=$BS  lr=$LR  epochs=$EPOCHS"
echo "   max_len  : $MAX_LEN   deepspeed=${DEEPSPEED_CFG:-<off, plain DDP>}"
echo "   out      : $OUT_DIR"
echo "================================================================="

"${LAUNCH[@]}" \
    --model_name_or_path "$BASE_MODEL" \
    --version "$CONV" \
    --mm_hidden_size 4096 \
    --mm_projector_type "$PROJECTOR_TYPE" \
    --tune_mm_mlp_adapter True \
    --mm_use_graph_start_end False \
    --mm_use_graph_patch_token False \
    --bf16 True \
    --tf32 True \
    --output_dir "$OUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BS" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --weight_decay "$WD" \
    --warmup_ratio "$WARMUP" \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --model_max_length "$MAX_LEN" \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --use_task nc \
    --use_dataset "$SOURCE" \
    --pretrained_embedding_type qwen3_emb \
    "${EXTRA_ARGS[@]}"

echo "[stage3] done. Projector: $OUT_DIR/mm_projector.bin"
