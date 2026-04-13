# Choose Your Own Anomaly (CYOA) Pipeline

An end-to-end active learning anomaly detection pipeline for time-domain astronomy (ZTF). It uses RTF autoencoders for rapid candidate filtering, Bayesian light curve fitting for physics extraction, and LLMs for real-time scientific triage and reporting.

> **SkyPortal UI Integration:** The frontend React widget (`CYOAWidget.jsx`) is being developed as a separate contribution to the main SkyPortal repository.
> **WIP Pull Request:** [skyportal/skyportal#6068](https://github.com/skyportal/skyportal/pull/6068)

---

## Pipeline Architecture & Execution

The pipeline is designed to run in an HPC environment (NCSA Delta) using SLURM. It executes 4 modular stages sequentially:

### Ingest (`ingest/`)
Connects to nightly ZTF archive tarballs from `ztf.uw.edu`, unpacks Avro files, and structures them into contiguous `alerts.npy` arrays for fast sequential reads.

**To run manually:**
```bash
sbatch ingest/slurm_unpack.sh 20260408
```

### Mass Inference (`inference/`)
Sweeps the RTF autoencoder over 150,000+ objects per night using multi-GPU batches. Calculates anomaly scores via Isolation Forest and reconstruction thresholding to flag the top 5% of transients.

**To run manually:**
```bash
sbatch inference/slurm_mass_infer.sh 20260408
```

### Precision Fitting (`precision_fitting/`)
Passes anomalous targets to the Rust-based Bayesian physical modeling backend (`boom-fit-batch`). Extracts thermal evolution, rise times, cooling rates, and standardized constraints (e.g. `dm15`, TDE decay power-law slopes).

**To run manually:**
```bash
sbatch precision_fitting/slurm_precision_fit.sh 20260408
```

### LLM Triage (`triage/`)
Transforms extracted physics parameters into expert-level scientific context prompts. Sends objects to an LLM (Llama-3.3-70B) for 10-class transient classification, confidence scoring, and scientific reasoning.

**To run manually:**
```bash
export GROQ_API_KEY="your_api_key_here"

python triage/triage.py \
    --json-dir /work/hdd/bcrv/kmajithia/sweep_results/20260408/applecider/json \
    --anomaly-csv /work/hdd/bcrv/kmajithia/sweep_results/20260408/threshold_anomalies.csv \
    --output /work/hdd/bcrv/kmajithia/sweep_results/20260408/triage_verdicts.jsonl \
    --api-key $GROQ_API_KEY \
    --top-n 50
```

### Active Learning (`active_learning/`)
Pushes vetted anomalies into the Fritz SkyPortal database via `POST /api/sources/{oid}/annotations`. Human reviewers provide feedback natively inside the Fritz interface. The `retrain.py` loop periodically sweeps those annotations to synthesize new training datasets (human clicks, LLM reviews, and statistical pseudo-labels) and update the per-group classifiers using `HistGradientBoosting`.

**To run manually:**
```bash
export FRITZ_TOKEN="your_fritz_token"

python active_learning/push_to_skyportal.py --verdicts triage_verdicts.jsonl
python active_learning/retrain.py --group-id 1947
```

---

## Fully Automated Nightly Run

Set your environment variables:
```bash
export GROQ_API_KEY="your_groq_key"
export FRITZ_TOKEN="your_fritz_token"
```

Then execute the master orchestrator for the current date:
```bash
chmod +x run_nightly_cron.sh
./run_nightly_cron.sh
```

This script acts as a SLURM dispatcher, waiting for each HPC job to complete before launching the next stage.

---

## Environment Setup

```bash
pip install numpy pandas scikit-learn pyarrow groq requests fastavro
```

Cargo (Rust) is required for compiling the AppleCiDEr fitting engine in the precision fitting stage.
