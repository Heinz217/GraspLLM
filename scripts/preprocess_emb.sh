set -euo pipefail

DATASET=${1:?"dataset name (e.g. cora, arxiv, computer)"}
GPUS=${2:-0}                                    # comma-sep ids, or single id
BATCH_SIZE=${BATCH_SIZE:-32}
MODEL_PATH=${QWEN3_EMB_MODEL:-}                 # optional override

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Count number of shards
NSHARDS=$(awk -F',' '{print NF}' <<<"$GPUS")

EXTRA_ARGS=()
[[ -n "$MODEL_PATH" ]] && EXTRA_ARGS+=(--model-path "$MODEL_PATH")

if [[ "$NSHARDS" -eq 1 ]]; then
    echo "[preprocess_emb] single-GPU run: dataset=$DATASET gpu=$GPUS bs=$BATCH_SIZE"
    CUDA_VISIBLE_DEVICES="$GPUS" python -u preprocess/build_qwen3_embeddings.py \
        --datasets "$DATASET" --batch-size "$BATCH_SIZE" "${EXTRA_ARGS[@]}"
else
    echo "[preprocess_emb] multi-GPU sharded: dataset=$DATASET gpus=$GPUS shards=$NSHARDS bs=$BATCH_SIZE"
    IFS=',' read -ra GPU_ARR <<<"$GPUS"
    PIDS=()
    for i in "${!GPU_ARR[@]}"; do
        gid="${GPU_ARR[$i]}"
        CUDA_VISIBLE_DEVICES="$gid" python -u preprocess/build_qwen3_embeddings.py \
            --datasets "$DATASET" \
            --batch-size "$BATCH_SIZE" \
            --shard-id "$i" --num-shards "$NSHARDS" \
            "${EXTRA_ARGS[@]}" &
        PIDS+=($!)
    done
    for pid in "${PIDS[@]}"; do wait "$pid"; done

    echo "[preprocess_emb] merging $NSHARDS shards"
    CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" python -u preprocess/build_qwen3_embeddings.py \
        --datasets "$DATASET" \
        --merge-shards --num-shards "$NSHARDS" "${EXTRA_ARGS[@]}"
fi

echo "[preprocess_emb] done — wrote ${GRASPLLM_DATASET_ROOT:-$REPO/dataset}/$DATASET/qwen3_emb_x.pt"
