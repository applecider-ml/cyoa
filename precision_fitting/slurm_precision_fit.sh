#!/bin/bash
#SBATCH --job-name=applecider
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/applecider_%j.log

# ─── Environment ─────────────────────────────────────────────────────────────
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONUNBUFFERED=1

# ─── Paths ───────────────────────────────────────────────────────────────────
ANOMALY_CSV="/work/hdd/bcrv/kmajithia/sweep_results/20260408/threshold_anomalies.csv"
NPY_DIR="/work/hdd/bcrv/kmajithia/uw_npy/20260408"
BINARY="/work/hdd/bcrv/kmajithia/boom-fitting-ml/target/release/boom-fit-batch"
OUTPUT_DIR="/work/hdd/bcrv/kmajithia/sweep_results/20260408/applecider"

mkdir -p "$OUTPUT_DIR"
mkdir -p /work/hdd/bcrv/kmajithia/logs

# ─── Build the Rust binary if not present ────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "[$(date)] Binary not found. Building boom-fit-batch..."
    cd /work/hdd/bcrv/kmajithia/boom-fitting-ml
    cargo build --release 2>&1
    echo "[$(date)] Build complete."
else
    echo "[$(date)] Binary found: $BINARY"
fi

echo "[$(date)] Starting Phase 5.3: Precision AppleCiDEr fitting"
echo "[$(date)] Anomaly CSV: $ANOMALY_CSV"
echo "[$(date)] NPY dir:     $NPY_DIR"
echo "[$(date)] Output dir:  $OUTPUT_DIR"

python /work/hdd/bcrv/kmajithia/boom-fitting-ml/run_precision_applecider.py \
    --anomaly-csv  "$ANOMALY_CSV" \
    --npy-dir      "$NPY_DIR" \
    --binary       "$BINARY" \
    --output-dir   "$OUTPUT_DIR" \
    --top-n        500 \
    --threads      16

echo "[$(date)] Phase 5.3 complete!"
echo "[$(date)] JSON results in: $OUTPUT_DIR/json/"
