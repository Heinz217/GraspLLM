set -euo pipefail

DATASET=${1:?"dataset name (e.g. cora, arxiv, computer)"}
shift || true

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

THRESHOLD=${THRESHOLD:-0.1}
BETA=${BETA:-0.55}
GPU=${GPU:-0}

echo "[stage2] dataset=$DATASET threshold=$THRESHOLD beta=$BETA gpu=$GPU args=$*"

cd "$REPO/gnn"

CUDA_VISIBLE_DEVICES="$GPU" python -u seq.py \
    --dataset "$DATASET" \
    --threshold "$THRESHOLD" \
    --beta "$BETA" \
    "$@"

echo "[stage2] done."
