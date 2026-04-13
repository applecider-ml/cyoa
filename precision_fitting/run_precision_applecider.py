"""
run_precision_applecider.py — Phase 5.3: Precision AppleCiDEr Fitting

Reads the top N anomalies from threshold_anomalies.csv (output of Phase 5.2),
extracts their photometry from alerts.npy files, and runs the boom-fit-batch
Rust binary to obtain full nonparametric + parametric + thermal fitting results.

The binary supports two input layouts:
  NPZ: {input_dir}/{split}/{obj_id}.npz
  CSV: {input_dir}/{obj_id}/photometry.csv   ← we use this one

We write a minimal photometry.csv per anomaly using the raw alerts.npy data,
then invoke boom-fit-batch which writes one {obj_id}.json per source.

Usage:
    python run_precision_applecider.py \\
        --anomaly-csv    /work/hdd/bcrv/kmajithia/sweep_results/20260408/threshold_anomalies.csv \\
        --npy-dir        /work/hdd/bcrv/kmajithia/uw_npy/20260408 \\
        --binary         /work/hdd/bcrv/kmajithia/boom-fitting-ml/target/release/boom-fit-batch \\
        --output-dir     /work/hdd/bcrv/kmajithia/sweep_results/20260408/applecider \\
        --top-n          500 \\
        --threads        16

Output:
    {output_dir}/{obj_id}.json  — one file per anomaly with:
        nonparametric: GP-based rise/fade/duration features
        parametric:    Bazin / Villar flux-model fits
        thermal:       Blackbody temperature + cooling_rate estimates
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


# ── Photometry extraction ─────────────────────────────────────────────────────

def extract_photometry_csv(alerts: np.ndarray, out_path: str) -> bool:
    """
    Extract a minimal photometry.csv from an alerts.npy array.
    The boom-fit-batch CSV reader expects columns: jd, magpsf, sigmapsf, fid
    Returns True if at least 5 valid historical rows were written.
    """
    rows = []
    
    def add_cand(cand):
        try:
            jd       = float(cand.get("jd"))
            magpsf   = float(cand.get("magpsf"))
            sigmapsf = float(cand.get("sigmapsf"))
            fid      = int(cand.get("fid"))
            
            # Filter non-detections and bogus limits
            if 15.0 < magpsf < 25.0 and 0 < sigmapsf <= 2.0:
                rows.append({"jd": jd, "magpsf": magpsf, "sigmapsf": sigmapsf, "fid": fid})
        except (KeyError, TypeError, ValueError):
            pass

    for a in alerts:
        # 1. Latest detection
        if "candidate" in a and a["candidate"]:
            add_cand(a["candidate"])
            
        # 2. Historical detections (UW packets bundle ~30 days of history here)
        if "prv_candidates" in a and a["prv_candidates"]:
            for p in a["prv_candidates"]:
                if p.get("magpsf") is not None:  # ignore upper limits where magpsf is null
                    add_cand(p)

    if len(rows) < 5:
        return False

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["jd", "magpsf", "sigmapsf", "fid"])
        writer.writeheader()
        writer.writerows(rows)

    return True


def prepare_photometry(oids: list[str], npy_dir: str, photo_dir: str) -> tuple[list[str], list[str]]:
    """
    For each OID, read alerts.npy and write photometry.csv to a flat staging
    directory that boom-fit-batch can read via its CSV fallback path:
        {photo_dir}/{obj_id}/photometry.csv
    Returns (ready_oids, skipped_oids).
    """
    ready, skipped = [], []
    for i, oid in enumerate(oids):
        npy_path = Path(npy_dir) / oid / "alerts.npy"
        out_dir  = Path(photo_dir) / oid
        out_path = out_dir / "photometry.csv"

        if out_path.exists():
            ready.append(oid)
            continue

        if not npy_path.exists():
            skipped.append(oid)
            continue

        try:
            alerts = np.load(npy_path, allow_pickle=True)
        except Exception:
            skipped.append(oid)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        ok = extract_photometry_csv(alerts, str(out_path))
        if ok:
            ready.append(oid)
        else:
            skipped.append(oid)

        if (i + 1) % 50 == 0:
            print(f"  Prepared {i + 1}/{len(oids)} — ready={len(ready)}, skipped={len(skipped)}")

    return ready, skipped


# ── Sources CSV writer ────────────────────────────────────────────────────────

def write_sources_csv(oids: list[str], path: str, split: str = "infer") -> None:
    """Write the obj_id,split CSV that boom-fit-batch --sources expects."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["obj_id", "split"])
        for oid in oids:
            writer.writerow([oid, split])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 5.3: Precision AppleCiDEr Fitting")
    parser.add_argument("--anomaly-csv", required=True,
                        help="threshold_anomalies.csv from Phase 5.2")
    parser.add_argument("--npy-dir", required=True,
                        help="Directory containing {obj_id}/alerts.npy")
    parser.add_argument("--binary", required=True,
                        help="Path to compiled boom-fit-batch binary")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write {obj_id}.json fitting results")
    parser.add_argument("--top-n", type=int, default=500,
                        help="Process only the top N anomalies by rank (default: 500)")
    parser.add_argument("--threads", type=int, default=16,
                        help="Rayon threads for the Rust binary (default: 16)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Read top N anomalies ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 5.3: Precision AppleCiDEr Fitting")
    print(f"Top-N      : {args.top_n}")
    print(f"Threads    : {args.threads}")
    print(f"{'='*60}\n")

    print(f"[1/4] Reading top {args.top_n} anomalies from {args.anomaly_csv}...")
    top_oids = []
    with open(args.anomaly_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top_oids.append(row["oid"])
            if len(top_oids) >= args.top_n:
                break

    print(f"  Loaded {len(top_oids)} OIDs (rank 1 to {len(top_oids)})")

    # ── 2. Extract photometry.csv for each anomaly ────────────────────────────
    photo_staging = os.path.join(args.output_dir, "_photometry_staging")
    print(f"\n[2/4] Extracting photometry from alerts.npy → {photo_staging}...")
    ready_oids, skipped_oids = prepare_photometry(top_oids, args.npy_dir, photo_staging)
    print(f"  Ready: {len(ready_oids):,}  |  Skipped (insufficient detections): {len(skipped_oids):,}")

    if not ready_oids:
        print("ERROR: No sources ready for fitting. Exiting.")
        sys.exit(1)

    # ── 3. Write sources CSV for the Rust binary ──────────────────────────────
    sources_csv = os.path.join(args.output_dir, "fit_sources.csv")
    print(f"\n[3/4] Writing sources CSV → {sources_csv}...")
    write_sources_csv(ready_oids, sources_csv, split="infer")
    print(f"  {len(ready_oids)} sources queued for fitting.")

    # ── 4. Run boom-fit-batch ─────────────────────────────────────────────────
    results_dir = os.path.join(args.output_dir, "json")
    os.makedirs(results_dir, exist_ok=True)

    cmd = [
        args.binary,
        "--input-dir",  photo_staging,
        "--sources",    sources_csv,
        "--output-dir", results_dir,
        "--threads",    str(args.threads),
    ]

    print(f"\n[4/4] Running boom-fit-batch on {len(ready_oids)} anomalies...")
    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nERROR: boom-fit-batch exited with code {result.returncode}")
        sys.exit(result.returncode)

    # ── Summary ───────────────────────────────────────────────────────────────
    json_files = list(Path(results_dir).glob("*.json"))
    print(f"\n{'='*60}")
    print(f"FITTING COMPLETE")
    print(f"  Sources submitted : {len(ready_oids):,}")
    print(f"  JSON files written: {len(json_files):,}")
    print(f"  Output dir        : {results_dir}")
    print(f"  Next step         : Phase 5.4 — LLM triage on {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
