#!/bin/bash
#SBATCH --job-name=rtf_sweep
#SBATCH --account=bcrv-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/rtf_sweep_%j.log

# ─── Environment ────────────────────────────────────────────────────────────
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONPATH=/work/hdd/bcrv/kmajithia/boom-fitting-ml:$PYTHONPATH
export PYTHONPATH=/work/hdd/bcrv/kmajithia/rtf/src:$PYTHONPATH
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1

# ─── GPU diagnostics ─────────────────────────────────────────────────────────
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi failed"

pip install scikit-learn --quiet   # ensure sklearn is available

# ─── Paths ───────────────────────────────────────────────────────────────────
NPY_DIR="/work/hdd/bcrv/kmajithia/uw_npy/20260408"
SPLITS_CSV="/work/hdd/bcrv/kmajithia/uw_npy/20260408/unlabelled_splits.csv"
# Best model from our Delta multi-modal RTF training run (128-dim, metadata + images)
MODEL_PATH="/work/hdd/bcrv/kmajithia/rtf/runs/ae_dim128_meta_img/best_model.pt"
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/sweep_results/20260408"

mkdir -p "$OUTPUT_DIR"
mkdir -p /work/hdd/bcrv/kmajithia/logs

echo "[$(date)] Starting RTF Mass Inference Sweep on A100 GPU"
echo "[$(date)] Input:  $NPY_DIR"
echo "[$(date)] Model:  $MODEL_PATH"
echo "[$(date)] Output: $OUTPUT_DIR"

python /work/hdd/bcrv/kmajithia/boom-fitting-ml/mass_infer_rtf.py \
    --npy-dir     "$NPY_DIR" \
    --splits-csv  "$SPLITS_CSV" \
    --model-path  "$MODEL_PATH" \
    --output-dir  "$OUTPUT_DIR" \
    --anomaly-percentile 95 \
    --batch-size  256 \
    --workers     0

echo "[$(date)] Sweep complete!"
echo "[$(date)] Anomaly CSV: $OUTPUT_DIR/threshold_anomalies.csv"
echo "[$(date)] Embeddings:  $OUTPUT_DIR/rtf_embeddings.npz"
