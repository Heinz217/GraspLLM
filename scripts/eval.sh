set -euo pipefail

CKPT=""; BACKBONE=""; DATASET=""; GPU=""; GPUS=""; TASK="nc"
TEST_PATH=""; BASE_MODEL=""; CONV_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt)      CKPT="$2";       shift 2;;
        --backbone)  BACKBONE="$2";   shift 2;;
        --dataset)   DATASET="$2";    shift 2;;
        --gpu)       GPU="$2";        shift 2;;
        --gpus)      GPUS="$2";       shift 2;;
        --task)      TASK="$2";       shift 2;;
        --test-path) TEST_PATH="$2";  shift 2;;
        --base-model) BASE_MODEL="$2"; shift 2;;
        --conv)      CONV_OVERRIDE="$2"; shift 2;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -n "$CKPT" ]]     || { echo "--ckpt required";     exit 1; }
[[ -n "$BACKBONE" ]] || { echo "--backbone required"; exit 1; }
[[ -n "$DATASET" ]]  || { echo "--dataset required";  exit 1; }

# Default: single GPU = 0 unless --gpus given
if [[ -z "$GPUS" ]]; then
    GPUS=${GPU:-0}
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# resolve backbone -> base model + conv template
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

# Default test_path: $GRASPLLM_DATASET_ROOT/<dataset>/ocs_test.jsonl
if [[ -z "$TEST_PATH" ]]; then
    TEST_PATH=$(python -c "from utils.paths import dataset_dir; import os; print(os.path.join(dataset_dir('$DATASET'), 'ocs_test.jsonl'))")
fi

[[ -f "$TEST_PATH" ]] || { echo "test set not found: $TEST_PATH"; exit 1; }

NSHARDS=$(awk -F',' '{print NF}' <<<"$GPUS")
TOTAL=$(wc -l < "$TEST_PATH")

run_eval_shard() {
    local gid="$1" start="$2" end="$3" ans_path="$4"
    CUDA_VISIBLE_DEVICES="$gid" python -u eval/eval_pretrain.py \
        --model_path "$CKPT" \
        --model_base "$BASE_MODEL" \
        --conv_mode "$CONV" \
        --dataset "$DATASET" \
        --task "$TASK" \
        --pretrained_embedding_type qwen3_emb \
        --answers_file "$ans_path" \
        --test_path "$TEST_PATH" \
        --start "$start" --end "$end" \
        --cache_dir "$CKPT/_eval_cache"
}

ANSWERS="$CKPT/answers_${DATASET}.jsonl"
rm -f "$ANSWERS"

if [[ "$NSHARDS" -eq 1 ]]; then
    echo "[eval] single-GPU on GPU=$GPUS  test_path=$TEST_PATH  N=$TOTAL"
    run_eval_shard "$GPUS" -1 -1 "$ANSWERS"
else
    echo "[eval] multi-GPU eval gpus=$GPUS shards=$NSHARDS  N=$TOTAL"
    IFS=',' read -ra GPU_ARR <<<"$GPUS"
    PIDS=()
    SHARD_FILES=()
    CHUNK=$(( (TOTAL + NSHARDS - 1) / NSHARDS ))
    for i in "${!GPU_ARR[@]}"; do
        gid="${GPU_ARR[$i]}"
        s=$(( i * CHUNK ))
        e=$(( s + CHUNK ))
        [[ "$e" -gt "$TOTAL" ]] && e="$TOTAL"
        f="$CKPT/answers_${DATASET}.shard${i}.jsonl"
        SHARD_FILES+=("$f")
        run_eval_shard "$gid" "$s" "$e" "$f" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    cat "${SHARD_FILES[@]}" > "$ANSWERS"
    rm -f "${SHARD_FILES[@]}"
    echo "[eval] merged shards -> $ANSWERS"
fi

# --------------- score ---------------
N=$(wc -l < "$ANSWERS")
echo "[eval] generated $N answers; computing metric ..."
python -u eval/eval_res.py \
    --res_path "$ANSWERS" \
    --task "$TASK" \
    --dataset "$DATASET"
