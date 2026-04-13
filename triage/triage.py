"""
triage.py -- Phase 5.4: LLM Scientific Triage of ZTF Anomalies

Reads the AppleCiDEr JSON fitting results for each anomaly, distills the key
physical observables into a compact prompt, and sends them to Groq's Llama-3.3-70B
for scientific classification and triage.

Each anomaly gets a structured JSON verdict containing:
  - primary_class:  most likely transient type (SN Ia, SLSN, TDE, CV, AGN, KN, Unknown)
  - confidence:     low / medium / high
  - reasoning:      2-3 sentence scientific justification
  - follow_up:      recommended follow-up action (trigger spectroscopy, monitor, archive)
  - flags:          list of interesting observational signatures

Usage:
    python triage.py \\
        --json-dir   /work/hdd/bcrv/kmajithia/sweep_results/20260408/applecider/json \\
        --anomaly-csv /work/hdd/bcrv/kmajithia/sweep_results/20260408/threshold_anomalies.csv \\
        --output     /work/hdd/bcrv/kmajithia/sweep_results/20260408/triage_verdicts.jsonl \\
        --api-key    $GROQ_API_KEY \\
        --top-n      50 \\
        --model      llama-3.3-70b-versatile

Requirements:
    pip install groq
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    print("ERROR: groq package not found. Run: pip install groq")
    sys.exit(1)


# ── Physical context to give the LLM ────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert time-domain astronomer specializing in transient classification for the Zwicky Transient Facility (ZTF). 

You will be given observational and physical features extracted from a ZTF light curve that has been flagged as anomalous by a deep learning autoencoder. Your task is to classify the transient and recommend follow-up action.

Known transient classes and their key signatures:
- SN Ia: rise ~15-20 days, decline ~30 days, dm15 ~ 0.8-1.6 mag, red near peak
- SN Ibc/II: faster or slower than Ia, bluer, various shapes
- SLSN: very slow rise (>30 days), extreme peak brightness (< -20 mag abs), blue
- TDE (Tidal Disruption Event): slow smooth rise weeks-months, power-law decay t^{-5/3} (decay_power_law_index ~ 4.2), blue color, von_neumann_ratio < 0.5
- CV (Cataclysmic Variable): rapid outbursts, multiple peaks, fast rise/fall < 5 days
- AGN: stochastic variability, high von_neumann_ratio ~ 1.5-2.0, no clear peak
- Kilonova: extremely fast rise < 1-2 days, rapid red evolution, faint
- Unknown: genuinely bizarre, does not fit known classes

Always respond ONLY with a valid JSON object. No prose, no markdown, no extra text. Just the JSON.
"""

TRIAGE_TEMPLATE = """Anomaly OID: {oid}
Rank: #{rank} out of 8868 flagged (rank 1 = most anomalous)
Combined anomaly score: {score:.4f} (reconstruction error + Isolation Forest)

--- PHOTOMETRIC FEATURES (r-band primary, g-band secondary) ---
Peak magnitude:          {peak_mag}
Rise time (halfmax):     {rise_halfmax} days
Rise time (e-fold):      {rise_efold} days  
Decay time (e-fold):     {decay_efold} days
Decay time (halfmax):    {decay_halfmax} days
dm15 (mag change@15d):   {dm15} mag
FWHM:                    {fwhm} days
Decay power-law index:   {decay_power_law_index} (TDE signature ~4.2)
Von Neumann ratio:       {von_neumann_ratio} (AGN~1.5-2.0, TDE<0.5)
N local maxima:          {n_local_maxima} (CV~2+, SN~1)
Post-peak monotonicity:  {post_peak_monotonicity} (TDE~1.0, AGN~0.5)
N observations:          {n_obs}

--- PARAMETRIC (Villar model SVI fit) ---
Model selected:          {model}
PSO chi2:                {pso_chi2}
Mag-space chi2:          {mag_chi2}

--- THERMAL ---
{thermal_summary}

Respond with a JSON object with exactly these keys:
{{
  "oid": "{oid}",
  "primary_class": "<class>",
  "confidence": "<low|medium|high>",
  "reasoning": "<2-3 sentences explaining key features that support your classification>",
  "follow_up": "<spectroscopy|photometric_monitoring|archive|rapid_spectroscopy>",
  "flags": ["<interesting feature 1>", "<interesting feature 2>"]
}}"""


# ── Feature extraction ───────────────────────────────────────────────────────


def safe_fmt(val, fmt=".2f", fallback="unknown"):
    """Format a float value safely, handling None."""
    if val is None:
        return fallback
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def extract_features(fitting: dict, oid: str, rank: int, score: float) -> str:
    """Build a triage prompt from an AppleCiDEr JSON fitting result."""
    nonparam = fitting.get("nonparametric", [])
    parametric = fitting.get("parametric", [])
    thermal = fitting.get("thermal")

    # Find r-band nonparametric result (primary band for ZTF)
    r_band = next((b for b in nonparam if b.get("band") == "r"), None)
    g_band = next((b for b in nonparam if b.get("band") == "g"), None)
    primary = r_band or g_band or (nonparam[0] if nonparam else {})

    # Parametric result
    param = parametric[0] if parametric else {}

    # Thermal summary
    if thermal:
        t_rise = thermal.get("temperature_at_peak")
        cooling = thermal.get("cooling_rate")
        thermal_summary = (
            f"Temperature at peak: {safe_fmt(t_rise, '.0f')} K  |  "
            f"Cooling rate: {safe_fmt(cooling, '.4f')} K/day"
        )
    else:
        thermal_summary = "No thermal fit available (insufficient multi-band coverage)"

    return TRIAGE_TEMPLATE.format(
        oid=oid,
        rank=rank,
        score=score,
        peak_mag=safe_fmt(primary.get("peak_mag"), ".3f"),
        rise_halfmax=safe_fmt(primary.get("rise_halfmax"), ".1f"),
        rise_efold=safe_fmt(primary.get("rise_efold"), ".1f"),
        decay_efold=safe_fmt(primary.get("decay_efold"), ".1f"),
        decay_halfmax=safe_fmt(primary.get("decay_halfmax"), ".1f"),
        dm15=safe_fmt(primary.get("dm15"), ".3f"),
        fwhm=safe_fmt(primary.get("fwhm"), ".1f"),
        decay_power_law_index=safe_fmt(primary.get("decay_power_law_index"), ".3f"),
        von_neumann_ratio=safe_fmt(primary.get("von_neumann_ratio"), ".3f"),
        n_local_maxima=primary.get("n_local_maxima", "unknown"),
        post_peak_monotonicity=safe_fmt(primary.get("post_peak_monotonicity"), ".3f"),
        n_obs=primary.get("n_obs", "unknown"),
        model=param.get("model", "unknown"),
        pso_chi2=safe_fmt(param.get("pso_chi2"), ".4f"),
        mag_chi2=safe_fmt(param.get("mag_chi2"), ".4f"),
        thermal_summary=thermal_summary,
    )


# ── Groq inference ───────────────────────────────────────────────────────────


def triage_anomaly(client: Groq, prompt: str, model: str, retries: int = 3) -> dict:
    """Send a single anomaly prompt to Groq and return the parsed JSON verdict."""
    for attempt in range(retries):
        try:
            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = chat.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt + 1}/{retries}: {e}")
                time.sleep(2**attempt)
            else:
                return {"error": str(e), "raw": ""}
    return {}


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.4: LLM Triage of ZTF Anomalies"
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing {obj_id}.json AppleCiDEr results",
    )
    parser.add_argument(
        "--anomaly-csv", required=True, help="threshold_anomalies.csv from Phase 5.2"
    )
    parser.add_argument(
        "--output", required=True, help="Output .jsonl file (one verdict per line)"
    )
    parser.add_argument("--api-key", required=True, help="Groq API key")
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Triage only the top N anomalies by rank (default: 50)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model to use (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (default: 0.5)",
    )
    args = parser.parse_args()

    client = Groq(api_key=args.api_key)
    json_dir = Path(args.json_dir)

    # 1. Read anomaly rankings from CSV
    print(f"\n{'=' * 60}")
    print("Phase 5.4: LLM Triage")
    print(f"Model      : {args.model}")
    print(f"Top-N      : {args.top_n}")
    print(f"{'=' * 60}\n")

    print(f"[1/3] Loading anomaly rankings from {args.anomaly_csv}...")
    anomaly_rows = []
    seen_oids = set()
    with open(args.anomaly_csv, newline="") as f:
        for row in csv.DictReader(f):
            oid = row["oid"]
            if oid in seen_oids:
                continue
            seen_oids.add(oid)
            anomaly_rows.append(row)
            if len(anomaly_rows) >= args.top_n:
                break
    print(f"  Loaded {len(anomaly_rows)} unique anomalies.")

    # 2. Match to JSON fitting results
    print("\n[2/3] Matching anomalies to AppleCiDEr fitting results...")
    matched = []
    for row in anomaly_rows:
        oid = row["oid"]
        json_path = json_dir / f"{oid}.json"
        if json_path.exists():
            try:
                with open(json_path) as f:
                    fitting = json.load(f)
                matched.append(
                    {
                        "oid": oid,
                        "rank": int(row["rank"]),
                        "score": float(row["combined_score"]),
                        "fitting": fitting,
                    }
                )
            except Exception as e:
                print(f"  WARN: Could not read {json_path}: {e}")
    print(
        f"  Matched: {len(matched)} / {len(anomaly_rows)} anomalies have fitting results."
    )

    if not matched:
        print("ERROR: No matched anomalies. Did Phase 5.3 complete successfully?")
        sys.exit(1)

    # 3. LLM triage
    print(f"\n[3/3] Running LLM triage on {len(matched)} anomalies...")
    os.makedirs(Path(args.output).parent, exist_ok=True)

    success, failed = 0, 0
    with open(args.output, "w") as out_f:
        for i, item in enumerate(matched):
            oid = item["oid"]
            print(
                f"  [{i + 1:3d}/{len(matched)}] {oid} (rank #{item['rank']})...",
                end=" ",
                flush=True,
            )

            prompt = extract_features(item["fitting"], oid, item["rank"], item["score"])
            verdict = triage_anomaly(client, prompt, args.model)

            # Augment with metadata
            verdict["_rank"] = item["rank"]
            verdict["_score"] = item["score"]
            verdict["_phase"] = "5.4"

            out_f.write(json.dumps(verdict) + "\n")
            out_f.flush()

            if "error" not in verdict:
                print(
                    f"{verdict.get('primary_class', '?')} ({verdict.get('confidence', '?')})"
                )
                success += 1
            else:
                print(f"ERROR: {verdict.get('error', 'unknown')}")
                failed += 1

            if args.delay > 0 and i < len(matched) - 1:
                time.sleep(args.delay)

    print(f"\n{'=' * 60}")
    print("TRIAGE COMPLETE")
    print(f"  Succeeded: {success}")
    print(f"  Failed:    {failed}")
    print(f"  Output:    {args.output}")
    print(f"{'=' * 60}")

    # Quick summary
    print("\nClassification Summary:")
    verdicts = []
    with open(args.output) as f:
        for line in f:
            try:
                verdicts.append(json.loads(line))
            except Exception:
                pass

    from collections import Counter

    classes = Counter(
        v.get("primary_class", "Unknown") for v in verdicts if "error" not in v
    )
    for cls, count in classes.most_common():
        print(f"  {cls:30s}: {count}")


if __name__ == "__main__":
    main()
