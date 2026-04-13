"""
avro_to_npy.py — Convert raw ZTF AVRO alert files into alerts.npy per-object format.

The UW Archive tarballs contain raw .avro files in a flat directory:
    ztf_avro/{YYYYMMDD}/ZTF{id}.avro

This script converts them into the {obj_id}/alerts.npy structure that
preprocess_alerts.py + the RTF model training pipeline expects:
    output_dir/{obj_id}/alerts.npy   → array of raw alert dicts

It also writes a companion file:
    output_dir/unlabelled_splits.csv → obj_id,split (all rows set to "infer")
    output_dir/unlabelled_labels/    → dummy label .npz (label=−1, meaning "unknown")

This bridging step allows the existing preprocess_alerts.py to run unchanged
over the unlabelled raw UW stream.

Usage:
    python avro_to_npy.py \
        --avro-dir /tmp/ztf_avro_extracted/ \
        --output-dir /work/hdd/bcrv/kmajithia/ztf_alerts_npy \
        --workers 16
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import fastavro
import numpy as np


def read_avro_alerts(avro_path):
    """Read all alert records from a single .avro file."""
    alerts = []
    with open(avro_path, "rb") as f:
        reader = fastavro.reader(f)
        for record in reader:
            alerts.append(record)
    return alerts


def process_avro_file(args):
    """Convert a single .avro file into {obj_id}/alerts.npy + dummy label."""
    avro_path, output_dir, labels_dir = args

    try:
        alerts = read_avro_alerts(avro_path)
    except Exception as e:
        return "skip", str(e)

    if not alerts:
        return "skip", "empty"

    # ZTF object ID is the candidate objectId field in the first alert
    obj_id = alerts[0].get("objectId", None)
    if obj_id is None:
        return "skip", "no objectId"

    # Write alerts.npy
    obj_dir = Path(output_dir) / obj_id
    obj_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = obj_dir / "alerts.npy"
    if not alerts_path.exists():
        np.save(str(alerts_path), np.array(alerts, dtype=object))

    # Write dummy label .npz (label=-1 means "unknown/unlabelled")
    label_dir = Path(labels_dir) / "infer"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{obj_id}.npz"
    if not label_path.exists():
        np.savez(str(label_path), label=np.int64(-1))

    return "done", obj_id


def main():
    parser = argparse.ArgumentParser(
        description="AVRO → alerts.npy converter for UW ZTF tarballs"
    )
    parser.add_argument(
        "--avro-dir",
        required=True,
        help="Path to extracted AVRO files (flat or nested)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root for {obj_id}/alerts.npy structure",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers"
    )
    args = parser.parse_args()

    avro_dir = Path(args.avro_dir)
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "unlabelled_labels"

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "infer").mkdir(parents=True, exist_ok=True)

    # Find all .avro files recursively (handles nested date subdirs)
    avro_files = sorted(avro_dir.rglob("*.avro"))
    print(f"Found {len(avro_files)} .avro files in {avro_dir}")

    if len(avro_files) == 0:
        print("No .avro files found. Check the extraction path.")
        return

    tasks = [(str(f), str(output_dir), str(labels_dir)) for f in avro_files]

    done_ids = []
    skipped = 0

    if args.workers <= 1:
        for i, task in enumerate(tasks):
            status, result = process_avro_file(task)
            if status == "done":
                done_ids.append(result)
            else:
                skipped += 1
            if (i + 1) % 1000 == 0:
                print(
                    f"  {i + 1}/{len(tasks)} — done={len(done_ids)}, skipped={skipped}"
                )
    else:
        with Pool(args.workers) as pool:
            for i, (status, result) in enumerate(
                pool.imap_unordered(process_avro_file, tasks, chunksize=50)
            ):
                if status == "done":
                    done_ids.append(result)
                else:
                    skipped += 1
                if (i + 1) % 1000 == 0:
                    print(
                        f"  {i + 1}/{len(tasks)} — done={len(done_ids)}, skipped={skipped}"
                    )

    print(f"\nConversion complete: {len(done_ids)} objects written, {skipped} skipped")

    # Write the splits CSV (all rows labelled "infer" since these are unlabelled unknowns)
    splits_path = output_dir / "unlabelled_splits.csv"
    with open(splits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["obj_id", "split"])
        writer.writeheader()
        for obj_id in done_ids:
            writer.writerow({"obj_id": obj_id, "split": "infer"})

    print(f"Splits CSV written to: {splits_path}")
    print(f"Dummy labels written to: {labels_dir}/infer/")
    print(f"\nNext step: run preprocess_alerts.py pointing at {output_dir}")


if __name__ == "__main__":
    main()
