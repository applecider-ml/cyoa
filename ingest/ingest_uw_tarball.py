"""
ingest_uw_tarball.py

Converts raw .avro alert files from a UW nightly tarball into unlabelled
PyTorch-compatible .npz tensors for RTF Autoencoder inference.

Unlike preprocess_alerts.py (which requires splits.csv + labels),
this script handles raw, unlabelled transients from the public UW archive.

Usage:
    python ingest_uw_tarball.py \
        --avro-dir /tmp/uw_alerts/ztf_public_20260408 \
        --output-dir /work/hdd/bcrv/kmajithia/uw_npz \
        --workers 16

Output:
    One .npz file per ZTF object ID containing:
        x:          (L, 37) float32 — photometry + metadata tensor (RTF input)
        images:     (L, 3, 63, 63) float32 — cutout stamps
        has_image:  (L,) float32 — 1.0 if cutout existed
        oid:        str — ZTF object ID
"""

import argparse
import gzip
import io
import os
from multiprocessing import Pool
from pathlib import Path

import fastavro
import numpy as np

# RTF metadata keys — must match dataset.py ALERT_META_KEYS exactly
ALERT_META_KEYS = [
    "ra", "dec", "magpsf", "sigmapsf", "chipsf", "magap", "sigmagap",
    "distnr", "magnr", "sigmagnr", "chinr", "sharpnr", "sky", "magdiff",
    "fwhm", "classtar", "mindtoedge", "magfromlim", "seeratio", "aimage",
    "bimage", "aimagerat", "bimagerat", "nneg", "nbad", "rb", "ssdistnr",
    "ssmagnr", "sumrat", "scorr",
]

STAMP_SIZE = 63
FID_TO_BAND = {1: "ztfg", 2: "ztfr", 3: "ztfi"}


def decode_stamp(stamp_bytes):
    """Decode a gzip-compressed FITS stamp to a numpy array."""
    from astropy.io import fits
    decompressed = gzip.decompress(stamp_bytes)
    with fits.open(io.BytesIO(decompressed), ignore_missing_simple=True) as hdu:
        return hdu[0].data.astype(np.float32)


def extract_cutouts(alert):
    """Extract (3, 63, 63) cutout array from a raw Avro alert dict."""
    try:
        sci = decode_stamp(alert["cutoutScience"]["stampData"])
        tmpl = decode_stamp(alert["cutoutTemplate"]["stampData"])
        diff = decode_stamp(alert["cutoutDifference"]["stampData"])
        if sci.shape != (STAMP_SIZE, STAMP_SIZE):
            return None
        if tmpl.shape != (STAMP_SIZE, STAMP_SIZE):
            return None
        if diff.shape != (STAMP_SIZE, STAMP_SIZE):
            return None
        sci = np.nan_to_num(sci, nan=0.0)
        tmpl = np.nan_to_num(tmpl, nan=0.0)
        diff = np.nan_to_num(diff, nan=0.0)
        return np.stack([sci, tmpl, diff], axis=0)
    except Exception:
        return None


def read_avro_alerts(avro_path):
    """Read all alert dicts from a single .avro file."""
    alerts = []
    with open(avro_path, "rb") as f:
        reader = fastavro.reader(f)
        for record in reader:
            alerts.append(record)
    return alerts


def build_tensor(alerts):
    """
    Convert a list of raw Avro alert dicts for ONE object into RTF input tensors.
    Returns (x, images, has_image) or None if the source is unusable.

    x:          (L, 37) float32
    images:     (L, 3, 63, 63) float32
    has_image:  (L,) float32
    """
    if not alerts:
        return None

    # Sort by JD
    try:
        jds = np.array([float(a["candidate"]["jd"]) for a in alerts])
    except Exception:
        return None
    order = np.argsort(jds)
    alerts = [alerts[i] for i in order]
    jds = jds[order]
    L = len(alerts)

    # Time features
    dt = (jds - jds[0]).astype(np.float32)
    dt_prev = np.zeros(L, dtype=np.float32)
    if L > 1:
        dt_prev[1:] = np.diff(jds).astype(np.float32)

    # Photometry
    try:
        fids = np.array([a["candidate"]["fid"] for a in alerts])
        magpsf = np.array([float(a["candidate"]["magpsf"]) for a in alerts], dtype=np.float32)
        sigmapsf = np.array([float(a["candidate"]["sigmapsf"]) for a in alerts], dtype=np.float32)
    except Exception:
        return None

    logflux = (-0.4 * magpsf).astype(np.float32)
    logflux_err = (0.4 * sigmapsf).astype(np.float32)
    log_dt = np.log1p(dt)
    log_dt_prev = np.log1p(dt_prev)

    band_idx = (fids - 1).clip(0, 2).astype(np.int64)
    one_hot = np.eye(3, dtype=np.float32)[band_idx]

    base = np.column_stack([log_dt, log_dt_prev, logflux, logflux_err])  # (L, 4)

    # Metadata
    meta = np.zeros((L, len(ALERT_META_KEYS)), dtype=np.float32)
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

    # Cutouts
    images = np.zeros((L, 3, STAMP_SIZE, STAMP_SIZE), dtype=np.float32)
    has_image = np.zeros(L, dtype=np.float32)
    for i, a in enumerate(alerts):
        cutout = extract_cutouts(a)
        if cutout is not None:
            images[i] = cutout
            has_image[i] = 1.0

    return x, images, has_image


def process_avro_file(task):
    """Worker: process a single .avro file → write one .npz per object ID."""
    avro_path, output_dir = task
    try:
        alerts = read_avro_alerts(avro_path)
    except Exception as e:
        return f"skip:{avro_path}:{e}"

    # Group alerts by objectId (one .avro file can contain multiple alerts
    # for the same object, but UW tarballs are typically one alert per file)
    by_oid = {}
    for alert in alerts:
        oid = alert.get("objectId", None)
        if oid is None:
            continue
        by_oid.setdefault(oid, []).append(alert)

    written = 0
    for oid, oid_alerts in by_oid.items():
        out_path = os.path.join(output_dir, f"{oid}.npz")
        if os.path.exists(out_path):
            # Merge new alerts into existing npz if this oid already appeared
            # in a previous .avro file from same night
            existing = dict(np.load(out_path, allow_pickle=True))
            result = build_tensor(oid_alerts)
            if result is None:
                continue
            x_new, img_new, hi_new = result
            x_merged = np.concatenate([existing["x"], x_new], axis=0)
            img_merged = np.concatenate([existing["images"], img_new], axis=0)
            hi_merged = np.concatenate([existing["has_image"], hi_new], axis=0)
            np.savez_compressed(out_path, x=x_merged, images=img_merged,
                                has_image=hi_merged, oid=np.bytes_(oid))
        else:
            result = build_tensor(oid_alerts)
            if result is None:
                continue
            x, images, has_image = result
            np.savez_compressed(out_path, x=x, images=images,
                                has_image=has_image, oid=np.bytes_(oid))
            written += 1

    return f"done:{written}"


def main():
    parser = argparse.ArgumentParser(
        description="Ingest raw UW ZTF tarball Avro alerts → unlabelled .npz tensors"
    )
    parser.add_argument("--avro-dir", required=True,
                        help="Directory containing extracted .avro files from the UW tarball")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for .npz files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    avro_files = sorted(Path(args.avro_dir).rglob("*.avro"))
    print(f"Found {len(avro_files)} .avro files in {args.avro_dir} (searched recursively)")

    tasks = [(str(p), args.output_dir) for p in avro_files]

    n_done = 0
    n_skip = 0
    if args.workers <= 1:
        for i, task in enumerate(tasks):
            result = process_avro_file(task)
            if result.startswith("done"):
                n_done += int(result.split(":")[1])
            else:
                n_skip += 1
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(tasks)} avro files processed...")
    else:
        with Pool(args.workers) as pool:
            for i, result in enumerate(
                pool.imap_unordered(process_avro_file, tasks, chunksize=50)
            ):
                if result.startswith("done"):
                    n_done += int(result.split(":")[1])
                else:
                    n_skip += 1
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(tasks)} avro files processed...")

    print(f"\nIngestion complete.")
    print(f"  Objects written: {n_done}")
    print(f"  Files skipped:   {n_skip}")
    out = Path(args.output_dir)
    print(f"  Total .npz files: {len(list(out.glob('*.npz')))}")


if __name__ == "__main__":
    main()
