# Choose Your Own Anomaly (CYOA) Pipeline

This repository contains an end-to-end active learning anomaly detection pipeline for time-domain astronomy (specifically the Zwicky Transient Facility). It leverages Representation-Conditioned Translate-and-Fill (RTF) autoencoders for rapid candidate filtering, Bayesian light curve fitting for physics extraction, and Large Language Models (LLMs) for real-time scientific triage and reporting.

---

## 🛠️ Pipeline Architecture & Execution

The pipeline is designed to run in an HPC environment (like NCSA Delta) using SLURM. It sequentially executes 5 modular stages:

### 1. Ingest (`1_ingest/`)
Connects to nightly ZTF archive tarballs, unpacks heavily nested Avro files, and structures them into contiguous `alerts.npy` arrays for fast sequential reads.

**To run manually:**
```bash
# Provide the YYYYMMDD date format you want to ingest
sbatch 1_ingest/slurm_unpack.sh 20260408
```

### 2. Mass Inference (`2_mass_inference/`)
Sweeps an RTF image-conditioned autoencoder over 150,000+ objects per night using multi-GPU batches. Calculates real-time anomaly scores via Isolation Forest and reconstruction thresholding to flag the top 5% of all transients.

**To run manually:**
```bash
# This requires a GPU partition (e.g., A100 or A40)
sbatch 2_mass_inference/slurm_mass_infer.sh 20260408
```

### 3. Precision Fitting (`3_precision_fitting/`)
Passes the anomalous targets to an integrated Rust-based Bayesian physical modeling backend (`boom-fit-batch`). Extracts thermal evolution, rise times, cooling rates, and standardized constraints (e.g. `dm15`, TDE decay power-law slopes).

**To run manually:**
```bash
# The Rust binary will compile automatically if not found
sbatch 3_precision_fitting/slurm_precision_fit.sh 20260408
```

### 4. LLM Triage (`4_llm_triage/`)
Transforms the extracted physics parameters into an expert-level scientific context prompt. Sends the objects to an LLM (e.g. Llama-3.3-70B) for 10-class transient classification, confidence scoring, and scientific argumentation.

**To run manually:**
```bash
source /path/to/your/env/bin/activate
pip install groq

export GROQ_API_KEY="your_api_key_here"

python 4_llm_triage/triage.py \
    --json-dir /work/hdd/bcrv/kmajithia/sweep_results/20260408/applecider/json \
    --anomaly-csv /work/hdd/bcrv/kmajithia/sweep_results/20260408/threshold_anomalies.csv \
    --output /work/hdd/bcrv/kmajithia/sweep_results/20260408/triage_verdicts.jsonl \
    --api-key $GROQ_API_KEY \
    --top-n 50
```

### 5. Integration & Active Learning (`5_active_learning/`)
Pushes the vetted anomalies to the UI layer using the REST API. Human reviewers inspect these targets and provide binary feedback tags natively inside the Fritz app. The `retrain.py` loop periodically retrieves these annotations to automatically synthesize new datasets (combining human clicks, LLM reviews, and statistical pseudo-labels) to update the Isolation Forest decision boundaries seamlessly using `HistGradientBoosting`.

### 6. SkyPortal React UI (`6_skyportal_ui_patch/`)
Contains the frontend `CYOAWidget.jsx` module designed to inject cleanly into the SkyPortal interface. This React component renders the AI astrophysics reasoning alongside the Active Learning triage buttons. 

## 🚀 Fully Automated Nightly Run

Instead of manually running each stage, you can orchestrate the entire pipeline end-to-end to run unattended. 

Set your environment variables:
```bash
export GROQ_API_KEY="your_groq_key"
```

Then simply execute the master orchestrator script for the current date:
```bash
chmod +x run_nightly_cron.sh
./run_nightly_cron.sh
```
This script acts as a master SLURM dispatcher, waiting for each HPC job to complete before launching the subsequent dependent Python scripts.

---

## Environment Setup
Ensure you have the required dependencies in your Python environment:
```bash
pip install numpy pandas scikit-learn pyarrow groq requests fastavro
```
Cargo (Rust) is required for compiling the AppleCiDEr fitting engine in Stage 3.
