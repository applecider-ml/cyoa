#!/bin/bash
# CYOA Master Orchestrator
# Example cron job script that executes the entire Choose Your Own Anomaly pipeline end-to-end.

# Exit immediately if a command exits with a non-zero status
set -e

DATE=$(date +"%Y%m%d")
echo "============================================================"
echo "Starting CYOA pipeline for $DATE"
echo "============================================================"

# Stage 1: Ingest UW Tarball
echo "[1/5] Ingesting UW Tarball..."
sbatch --wait 1_ingest/slurm_unpack.sh $DATE

# Stage 2: Mass Inference
echo "[2/5] Running RTF Mass Inference Sweep..."
sbatch --wait 2_mass_inference/slurm_mass_infer.sh $DATE

# Stage 3: AppleCiDEr Physics Extraction
echo "[3/5] Running Precision Bayesian Fitting..."
sbatch --wait 3_precision_fitting/slurm_precision_fit.sh $DATE

# Stage 4: LLM Triage
echo "[4/5] Running LLM Triage..."
# Activate poetry / conda environment with groq installed
source /path/to/your/env/bin/activate
python 4_llm_triage/triage.py \
    --json-dir /work/hdd/bcrv/kmajithia/sweep_results/$DATE/applecider/json \
    --anomaly-csv /work/hdd/bcrv/kmajithia/sweep_results/$DATE/threshold_anomalies.csv \
    --output /work/hdd/bcrv/kmajithia/sweep_results/$DATE/triage_verdicts.jsonl \
    --api-key $GROQ_API_KEY \
    --top-n 50

# Stage 5: UI Integration pending team confirmation
echo "[5/5] UI Integration Layer (Under Construction)..."

echo "============================================================"
echo "Pipeline complete for $DATE!"
echo "============================================================"
