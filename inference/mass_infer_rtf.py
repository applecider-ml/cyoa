"""
mass_infer_rtf.py — Phase 5.2: RTF Mass Inference Sweep

Reads DIRECTLY from {obj_id}/alerts.npy files — no preprocessing step needed.
Uses a single directory listing instead of 477k individual Lustre stat() calls.

Loads the frozen LightCurveCompressor (RTF Autoencoder, mode=ae) trained on
Delta, runs ALL 159k ZTF objects through it, extracts:
  1. 128-dim latent embedding via model.embed()
  2. Per-sample reconstruction error via model.reconstruction_error()
     → reconstruction error IS our primary anomaly signal.

Then runs Isolation Forest on the embeddings for a complementary anomaly score.
Combines both and writes every object above the threshold to threshold_anomalies.csv.

Usage:
    python mass_infer_rtf.py \
        --npy-dir /work/hdd/bcrv/kmajithia/uw_npy/20260408 \
        --model-path /work/hdd/bcrv/kmajithia/rtf/runs/ae_128/best_model.pt \
        --output-dir /work/hdd/bcrv/kmajithia/sweep_results/20260408 \
        --anomaly-percentile 95

Output:
    rtf_embeddings.npz        — (N, 128) embeddings + OIDs + recon errors
    threshold_anomalies.csv   — ALL objects above the percentile threshold, ranked
"""

import argparse
import gzip
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, Dataset

# ── Add RTF src to path ───────────────────────────────────────────────────────
RTF_SRC = Path(__file__).resolve().parent.parent / "rtf" / "src"
sys.path.insert(0, str(RTF_SRC))

from model import LightCurveCompressor

# ── Metadata keys — MUST match dataset.py ALERT_META_KEYS exactly ────────────
ALERT_META_KEYS = [
    "sgscore1", "sgscore2", "distpsnr1", "distpsnr2", "nmtchps",
    "sharpnr", "scorr", "diffmaglim", "sky", "ndethist", "ncovhist",
    "sigmapsf", "chinr", "classtar", "rb", "chipsf", "distnr", "magnr",
    "fwhm", "srmag1", "sgmag1", "simag1", "szmag1",
    "srmag2", "sgmag2", "simag2", "szmag2",
    "clrcoeff", "clrcounc", "zpclrcov",
]
N_META = len(ALERT_META_KEYS)          # 30
IN_CHANNELS = 4 + 3 + N_META          # 37  (must match trained model)
STAMP_SIZE = 63
MAX_LEN = 257                          # matches model default


# ─── Direct Dataset — ONE listing, no per-file stat() calls ─────────────────

class DirectAlertDataset(Dataset):
    """
    Reads {obj_id}/alerts.npy directly.
    OID list comes from the splits CSV written by avro_to_npy.py —
    ZERO Lustre stat() calls at init time. Paths are constructed directly.
    """

    def __init__(self, npy_dir: str, splits_csv: str, max_len: int = MAX_LEN):
        self.root = Path(npy_dir)
        self.max_len = max_len

        # Read OID list from CSV — single sequential read, no directory scan at all
        import csv
        with open(splits_csv, newline="") as f:
            reader = csv.DictReader(f)
            self.entries = [
                (row["obj_id"], str(self.root / row["obj_id"] / "alerts.npy"))
                for row in reader
            ]
        print(f"  DirectAlertDataset: {len(self.entries):,} objects from {splits_csv}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        oid, npy_path = self.entries[idx]
        try:
            alerts = np.load(npy_path, allow_pickle=True)
        except Exception:
            return None

        if alerts.ndim == 0 or len(alerts) == 0:
            return None

        # Sort by JD
        try:
            jds = np.array([float(a["candidate"]["jd"]) for a in alerts])
        except Exception:
            return None
        order = np.argsort(jds)
        alerts = alerts[order]
        jds = jds[order]
        L = min(len(alerts), self.max_len)
        alerts = alerts[:L]
        jds = jds[:L]

        # ── Time features ─────────────────────────────────────────────────────
        dt = (jds - jds[0]).astype(np.float32)
        dt_prev = np.zeros(L, dtype=np.float32)
        if L > 1:
            dt_prev[1:] = np.diff(jds).astype(np.float32)
        log_dt = np.log1p(dt)
        log_dt_prev = np.log1p(dt_prev)

        # ── Photometry ────────────────────────────────────────────────────────
        try:
            fids = np.array([a["candidate"]["fid"] for a in alerts])
            magpsf = np.array([float(a["candidate"]["magpsf"]) for a in alerts], dtype=np.float32)
            sigmapsf = np.array([float(a["candidate"]["sigmapsf"]) for a in alerts], dtype=np.float32)
        except Exception:
            return None

        logflux = (-0.4 * magpsf)
        logflux_err = (0.4 * sigmapsf)
        band_idx = (fids - 1).clip(0, 2).astype(np.int64)
        one_hot = np.eye(3, dtype=np.float32)[band_idx]
        base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])

        # ── Metadata ──────────────────────────────────────────────────────────
        meta = np.zeros((L, N_META), dtype=np.float32)
        for i, a in enumerate(alerts):
            cand = a["candidate"]
            for j, key in enumerate(ALERT_META_KEYS):
                val = cand.get(key, np.nan)
                if val is None or (isinstance(val, (int, float)) and val == -999):
                    val = np.nan
                try:
                    meta[i, j] = float(val)
                except (ValueError, TypeError):
                    meta[i, j] = np.nan
        meta = np.nan_to_num(meta, nan=0.0)

        x = np.concatenate([base, one_hot, meta], axis=1).astype(np.float32)  # (L, 37)

        # ── Latest cutout stamp (single image per source) ─────────────────────
        image = np.zeros((3, STAMP_SIZE, STAMP_SIZE), dtype=np.float32)
        for a in reversed(alerts):
            try:
                sci = self._decode_stamp(a["cutoutScience"]["stampData"])
                tmpl = self._decode_stamp(a["cutoutTemplate"]["stampData"])
                diff = self._decode_stamp(a["cutoutDifference"]["stampData"])
                if sci.shape == (STAMP_SIZE, STAMP_SIZE):
                    image = np.stack([
                        np.nan_to_num(sci), np.nan_to_num(tmpl), np.nan_to_num(diff)
                    ], axis=0)
                    break
            except Exception:
                continue

        return oid, x, image

    @staticmethod
    def _decode_stamp(stamp_bytes):
        from astropy.io import fits
        decompressed = gzip.decompress(stamp_bytes)
        with fits.open(io.BytesIO(decompressed), ignore_missing_simple=True) as hdu:
            return hdu[0].data.astype(np.float32)


def collate_fn(batch):
    """Pads variable-length sequences, creates pad_mask."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    oids, xs, images = zip(*batch)
    lengths = [len(x) for x in xs]
    max_L = max(lengths)

    B = len(batch)
    x_pad = torch.zeros(B, max_L, IN_CHANNELS)
    img_t = torch.zeros(B, 3, STAMP_SIZE, STAMP_SIZE)
    pad_mask = torch.ones(B, max_L, dtype=torch.bool)   # True = padding

    for i, (x, img) in enumerate(zip(xs, images)):
        L = len(x)
        x_pad[i, :L] = torch.from_numpy(x)
        img_t[i] = torch.from_numpy(img)
        pad_mask[i, :L] = False                          # False = valid

    return oids, x_pad, pad_mask, img_t


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device) -> LightCurveCompressor:
    """
    Load frozen RTF Autoencoder.
    Trained with: mode=ae, latent_dim=128, in_channels=37, d_model=256
    (see slurm_train_meta_delta.sh)
    """
    model = LightCurveCompressor(
        mode="ae",
        latent_dim=128,
        in_channels=IN_CHANNELS,   # 37
        d_model=256,
        n_heads=8,
        enc_layers=4,
        dec_layers=2,
        d_ff=512,
        dropout=0.3,
        use_images=True,
        image_backbone="simple",
    ).to(device)

    # Checkpoint is raw state_dict (not nested in a wrapper dict)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded frozen RTF model → {model_path}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model


# ── Main sweep ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5.2: RTF Mass Inference Sweep")
    parser.add_argument("--npy-dir", required=True,
                        help="Directory containing {obj_id}/alerts.npy (uw_npy/YYYYMMDD)")
    parser.add_argument("--splits-csv", required=True,
                        help="Path to unlabelled_splits.csv written by avro_to_npy.py")
    parser.add_argument("--model-path", required=True,
                        help="Path to best_model.pt from RTF training")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for rtf_embeddings.npz + threshold_anomalies.csv")
    parser.add_argument("--anomaly-percentile", type=float, default=95.0,
                        help="Keep top X%% most anomalous objects (default: top 5%%)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Test CUDA is actually usable — on shared GPU nodes SLURM may report CUDA
    # available but the device context is inaccessible (conflict with other jobs).
    if torch.cuda.is_available():
        try:
            torch.tensor([1.0]).cuda()    # real usability test
            device = torch.device("cuda")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        except RuntimeError as e:
            print(f"  WARNING: CUDA available but unusable ({e}). Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"RTF Mass Inference Sweep — Phase 5.2")
    print(f"Device     : {device}")
    print(f"Input dir  : {args.npy_dir}")
    print(f"Threshold  : top {100 - args.anomaly_percentile:.1f}% most anomalous")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load the frozen model ──────────────────────────────────────────────
    print("[1/4] Loading frozen RTF Autoencoder...")
    model = load_model(args.model_path, device)

    # ── 2. Build dataset with ONE directory listing ───────────────────────────
    print(f"\n[2/4] Building dataset from splits CSV (zero stat() calls)...")
    dataset = DirectAlertDataset(args.npy_dir, args.splits_csv)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # ── 3. The sweep ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Encoding {len(dataset):,} objects through RTF Autoencoder...")
    all_oids, all_embeddings, all_recon_errors = [], [], []
    processed = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            oids, x, pad_mask, images = batch
            x = x.to(device)
            pad_mask = pad_mask.to(device)
            images = images.to(device)

            embeddings = model.embed(x, pad_mask, images)           # (B, 128)
            recon_err = model.reconstruction_error(x, pad_mask, images)  # (B,)

            all_oids.extend(oids)
            all_embeddings.append(embeddings.cpu().numpy())
            all_recon_errors.append(recon_err.cpu().numpy())

            processed += x.shape[0]
            if processed // 5000 > (processed - x.shape[0]) // 5000:
                print(f"  {processed:,} / {len(dataset):,} objects processed...")

    embeddings = np.concatenate(all_embeddings, axis=0)
    recon_errors = np.concatenate(all_recon_errors, axis=0)
    print(f"  Sweep complete. Shape: {embeddings.shape}")

    # Save full matrix
    emb_path = os.path.join(args.output_dir, "rtf_embeddings.npz")
    np.savez_compressed(emb_path,
                        embeddings=embeddings,
                        recon_errors=recon_errors,
                        oids=np.array(all_oids, dtype=str))
    print(f"  Embeddings saved → {emb_path}")

    # ── 4. Anomaly scoring ────────────────────────────────────────────────────
    print(f"\n[4/4] Running Isolation Forest on {len(all_oids):,} embeddings...")
    iso = IsolationForest(n_estimators=200, contamination="auto",
                          random_state=42, n_jobs=-1)
    iso.fit(embeddings)
    if_raw = iso.score_samples(embeddings)   # lower = more anomalous

    # Normalize both signals to [0, 1] using NumPy 2.0 compatible ptp()
    recon_norm = (recon_errors - recon_errors.min()) / (np.ptp(recon_errors) + 1e-9)
    if_norm = (-if_raw - (-if_raw).min()) / (np.ptp(-if_raw) + 1e-9)

    # Combined: 60% reconstruction error + 40% Isolation Forest
    combined = 0.6 * recon_norm + 0.4 * if_norm

    # Threshold: keep everything above the percentile
    threshold_val = np.percentile(combined, args.anomaly_percentile)
    mask = combined >= threshold_val
    n_flagged = mask.sum()

    print(f"  Percentile threshold ({args.anomaly_percentile:.0f}th): {threshold_val:.4f}")
    print(f"  Objects flagged: {n_flagged:,} / {len(all_oids):,}")

    # Write CSV — sorted by combined score descending
    oids_arr = np.array(all_oids)
    flagged_idx = np.where(mask)[0]
    flagged_idx = flagged_idx[np.argsort(combined[flagged_idx])[::-1]]

    csv_path = os.path.join(args.output_dir, "threshold_anomalies.csv")
    with open(csv_path, "w") as f:
        f.write("rank,oid,combined_score,recon_error,if_score\n")
        for rank, idx in enumerate(flagged_idx, 1):
            f.write(f"{rank},{oids_arr[idx]},{combined[idx]:.6f},"
                    f"{recon_errors[idx]:.6f},{if_raw[idx]:.6f}\n")

    print(f"  Threshold CSV saved → {csv_path}")
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"  Processed  : {len(all_oids):,} objects")
    print(f"  Flagged    : {n_flagged:,} anomalies")
    print(f"  Next step  : Phase 5.3 — run AppleCiDEr on {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
