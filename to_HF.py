"""
to_HF.py — Upload trained CYOA models to HuggingFace Hub.

Run this ONCE from Delta after training completes to make the pipeline
fully reproducible for the entire team.

Usage (from Delta):
    python to_HF.py \
        --model-dir /work/hdd/bcrv/kmajithia/rtf/runs/ae_dim64/ \
        --repo applecider-ml/cyoa-models \
        --token $HF_TOKEN

Prerequisites:
    pip install huggingface-hub
    huggingface-cli login   (or pass --token)
"""

import os
import json
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
    raise SystemExit(1)

DEFAULT_REPO = "applecider-ml/cyoa-models"

# Files to upload from the model directory
UPLOAD_MANIFEST = [
    "best_model.pt",  # RTF Autoencoder checkpoint
    "isolation_forest.pkl",  # Fitted Isolation Forest (if present)
    "summary.json",  # Training metrics & hyperparameters
]


def create_model_card(model_dir: str) -> str:
    """Generate a HuggingFace model card with training metadata."""
    summary_path = os.path.join(model_dir, "summary.json")
    meta = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            meta = json.load(f)

    card = f"""---
tags:
  - astronomy
  - ztf
  - anomaly-detection
  - transient-classification
license: mit
---

# CYOA (Choose Your Own Anomaly) Models

Pre-trained models for the ZTF anomaly detection pipeline.

## RTF Transformer Autoencoder
- **Architecture:** Transformer AE, latent_dim={meta.get("latent_dim", 128)}
- **Parameters:** {meta.get("n_params", "N/A")}
- **Training data:** {meta.get("n_train", "N/A")} ZTF light curves
- **Best epoch:** {meta.get("best_epoch", "N/A")}
- **Band accuracy:** {meta.get("band_acc", "N/A")}

## Isolation Forest
- Trained on RTF latent embeddings
- contamination=0.05, n_estimators=200

## Usage
```python
from from_HF import download_models, verify_models
download_models()
verify_models()
```
"""
    return card


def upload_models(model_dir: str, repo_id: str, token: str = None):
    """Upload model artifacts to HuggingFace Hub."""
    api = HfApi()
    model_dir = Path(model_dir)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True, token=token)
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Warning creating repo: {e}")

    # Upload each file in the manifest
    uploaded = 0
    for filename in UPLOAD_MANIFEST:
        filepath = model_dir / filename
        if filepath.exists():
            print(f"Uploading {filename} ({filepath.stat().st_size / 1024:.1f} KB)...")
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=filename,
                repo_id=repo_id,
                token=token,
            )
            uploaded += 1
        else:
            print(f"Skipping {filename} (not found in {model_dir})")

    # Upload the model card
    card_content = create_model_card(str(model_dir))
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )

    print(
        f"\nUpload complete: {uploaded} files pushed to https://huggingface.co/{repo_id}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload trained CYOA models to HuggingFace Hub"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the directory containing trained model files",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"HuggingFace repo slug (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--token", default=None, help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print(
            "Warning: No HF token provided. You may need to run: huggingface-cli login"
        )

    upload_models(args.model_dir, args.repo, token)
