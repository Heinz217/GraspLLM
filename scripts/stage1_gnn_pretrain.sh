set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- config (edit / override via env) ---
DATASETS=${DATASETS:-"arxiv pubmed computer history reddit"}
SAMPLES_PER_DATASET=${SAMPLES_PER_DATASET:-60}
NUM_EPOCHS=${NUM_EPOCHS:-300}
LR=${LR:-1e-4}
NUM_SAMPLES=${NUM_SAMPLES:-2000}
SAMPLING_METHOD=${SAMPLING_METHOD:-n-hop}
N_HOP=${N_HOP:-2}
SHARED_DIM=${SHARED_DIM:-256}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-256}
OUT_CHANNELS=${OUT_CHANNELS:-128}
TAU=${TAU:-0.4}
GPU=${GPU:-0}
SEED=${SEED:-0}

CHECKPOINT_ROOT=${GRASPLLM_CHECKPOINT_ROOT:-$REPO/checkpoints}
mkdir -p "$CHECKPOINT_ROOT"
MODEL_SAVE_PATH=${MODEL_SAVE_PATH:-"$CHECKPOINT_ROOT/structure_learner_qwen3.pth"}

echo "================================================================="
echo " Stage-1 GNN pre-train"
echo " datasets : $DATASETS"
echo " epochs   : $NUM_EPOCHS  lr=$LR  num_samples=$NUM_SAMPLES"
echo " gpu      : $GPU"
echo " save     : $MODEL_SAVE_PATH"
echo "================================================================="

cd "$REPO/gnn"

CUDA_VISIBLE_DEVICES="$GPU" python -u train.py \
    --datasets $DATASETS \
    --samples-per-dataset "$SAMPLES_PER_DATASET" \
    --num-epochs "$NUM_EPOCHS" \
    --lr "$LR" \
    --num-samples "$NUM_SAMPLES" \
    --sampling-method "$SAMPLING_METHOD" \
    --n-hop "$N_HOP" \
    --shared-dim "$SHARED_DIM" \
    --hidden-channels "$HIDDEN_CHANNELS" \
    --out-channels "$OUT_CHANNELS" \
    --tau "$TAU" \
    --device cuda \
    --seed "$SEED" \
    --model-save-path "$MODEL_SAVE_PATH"

echo
echo "[stage1] done. Checkpoint: $MODEL_SAVE_PATH"
