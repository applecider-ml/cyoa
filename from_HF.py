"""
from_HF.py — Download CYOA pre-trained models from HuggingFace Hub.

Mirrors the BTSbot pattern (https://github.com/nabeelre/BTSbot).
When a researcher clones the cyoa repo and runs the pipeline for the first
time, this script automatically pulls the trained artifacts so they never
need to manually copy .pt files from Delta.

Usage:
    python from_HF.py
    python from_HF.py --repo applecider-ml/cyoa-models --dest models/
"""

import os
import argparse

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
    raise SystemExit(1)

DEFAULT_REPO = "applecider-ml/cyoa-models"
DEFAULT_DEST = os.path.join(os.path.dirname(__file__), "models")

# Expected artifacts after download
EXPECTED_FILES = [
    "best_model.pt",          # RTF Transformer Autoencoder checkpoint
    "isolation_forest.pkl",   # Fitted scikit-learn Isolation Forest
    "train_config.json",      # Model hyperparameters & metadata
]


def download_models(repo_id: str = DEFAULT_REPO, local_dir: str = DEFAULT_DEST):
    """Download all CYOA model artifacts from HuggingFace Hub."""
    os.makedirs(local_dir, exist_ok=True)

    # Check if already present
    if all(os.path.isfile(os.path.join(local_dir, f)) for f in EXPECTED_FILES):
        print(f"All model files already present in {local_dir}. Skipping download.")
        return local_dir

    print(f"Downloading CYOA models from HuggingFace: {repo_id}")
    print(f"Destination: {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
    )

    print(f"Download complete. Models saved to {local_dir}")
    return local_dir


def verify_models(local_dir: str = DEFAULT_DEST):
    """Verify that all expected model files are present after download."""
    missing = [f for f in EXPECTED_FILES if not os.path.isfile(os.path.join(local_dir, f))]
    if missing:
        print(f"WARNING: Missing model files: {missing}")
        return False
    print("All model files verified successfully.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download CYOA pre-trained models from HuggingFace Hub"
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO,
        help=f"HuggingFace repo slug (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--dest", default=DEFAULT_DEST,
        help=f"Local download directory (default: {DEFAULT_DEST})"
    )
    args = parser.parse_args()

    download_models(args.repo, args.dest)
    verify_models(args.dest)
