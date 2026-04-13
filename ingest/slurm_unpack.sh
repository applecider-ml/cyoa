#!/bin/bash
#SBATCH --job-name=uw_unpack_ingest
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/work/hdd/bcrv/kmajithia/logs/unpack_%j.log

# ─── Environment ────────────────────────────────────────────────────────────
source /work/hdd/bcrv/kmajithia/envs/rtf/bin/activate
export PYTHONPATH=/work/hdd/bcrv/kmajithia/boom-fitting-ml:$PYTHONPATH

# Install fastavro if not already present in the rtf environment
pip install fastavro --quiet

# ─── Paths ───────────────────────────────────────────────────────────────────
TARBALL="/work/hdd/bcrv/kmajithia/uw_archive/ztf_public_20260408.tar.gz"
AVRO_DIR="/tmp/uw_alerts_20260408"        # Node-local NVMe: fast I/O, auto-cleaned
NPY_DIR="/work/hdd/bcrv/kmajithia/uw_npy/20260408"

mkdir -p "$AVRO_DIR"
mkdir -p "$NPY_DIR"
mkdir -p /work/hdd/bcrv/kmajithia/logs

# ─── Step 1: Extract tarball to node-local NVMe (flat layout — no subdirs) ──
# NOTE: No --strip-components here. UW tarballs are flat: just *.avro at root.
echo "[$(date)] Extracting tarball to node-local NVMe: $AVRO_DIR"
tar -xzf "$TARBALL" -C "$AVRO_DIR"
AVRO_COUNT=$(find "$AVRO_DIR" -name "*.avro" | wc -l)
echo "[$(date)] Extraction complete. $AVRO_COUNT .avro files found."

# ─── Step 2: Convert raw .avro → {obj_id}/alerts.npy + dummy labels ─────────
# avro_to_npy.py reads the real objectId from inside each avro file,
# creates the per-object directory structure, and generates unlabelled_splits.csv
# so that preprocess_alerts.py can run unchanged in Step 3.
echo "[$(date)] Starting Avro → alerts.npy conversion with 16 workers..."
python /work/hdd/bcrv/kmajithia/boom-fitting-ml/avro_to_npy.py \
    --avro-dir "$AVRO_DIR" \
    --output-dir "$NPY_DIR" \
    --workers 16

echo "[$(date)] Conversion complete."
echo "[$(date)] Unique ZTF objects: $(ls $NPY_DIR | grep -v 'unlabelled' | wc -l)"

