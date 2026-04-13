"""
Phase 6.0 Backend: Push to SkyPortal (Fritz)

This script parses the LLM AI verdicts from `triage_verdicts.jsonl`.
It performs a 2-step pipeline per anomaly:
1. Hit custom `/api/boom/alert` to compel Fritz to fetch & natively ingest the light curves.
2. Hit `/api/annotation` to push the LLM's classification & physical reasoning, scoped to the specific group.
"""

import argparse
import json
import time
import requests
from pathlib import Path

# Fritz API Configuration
FRITZ_URL = "https://fritz.science"

def push_anomaly_to_fritz(oid: str, survey: str, group_id: int, token: str) -> bool:
    """
    Step 1: Triggers the custom BOOM alert endpoint on Fritz.
    """
    url = f"{FRITZ_URL}/api/boom/alerts/{survey}/{oid}"
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "group_ids": [int(group_id)]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print(f"  [STEP 1 SUCCESS] Pushed {oid} to SkyPortal Group {group_id}.")
            return True
        else:
            print(f"  [STEP 1 ERROR] Failed to push {oid}. Code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  [STEP 1 NETWORK ERROR] {e}")
        return False

def push_annotation_to_fritz(oid: str, verdict: dict, group_id: int, token: str) -> bool:
    """
    Step 2: Pushes the explicit LLM reasoning into SkyPortal's Annotation schema.
    This provides the mathematical/physical explanation to the frontend React widget.
    """
    url = f"{FRITZ_URL}/api/sources/{oid}/annotations"
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json"
    }
    
    # Bundle the LLM intelligence — keys match triage.py output schema exactly
    annotation_data = {
        "cyoa_ai_class":   verdict.get("primary_class", "Unknown"),
        "cyoa_confidence": verdict.get("confidence", "medium"),
        "cyoa_reasoning":  verdict.get("reasoning", "No reasoning provided."),
        "cyoa_follow_up":  verdict.get("follow_up", "archive"),
        "cyoa_flags":      verdict.get("flags", []),
        "cyoa_rank":       verdict.get("_rank"),
        "cyoa_score":      verdict.get("_score"),
    }
    
    payload = {
        "obj_id": oid,
        "origin": "CYOA-Llama-3.3",
        "data": annotation_data,
        "group_ids": [int(group_id)]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        # 200 OK or 400 if annotation already exists (which we can catch)
        if response.status_code in [200, 400]:
            print(f"  [STEP 2 SUCCESS] Injected LLM reasoning for {oid}.")
            return True
        else:
            print(f"  [STEP 2 ERROR] Failed Annotation for {oid}. Code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  [STEP 2 NETWORK ERROR] {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Push CYOA anomalies & LLM annotations to SkyPortal UI")
    parser.add_argument("--input", required=True, help="Path to triage_verdicts.jsonl")
    parser.add_argument("--token", required=True, help="Fritz SkyPortal API Token")
    parser.add_argument("--group-id", required=True, help="Fritz Group ID to push objects to")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Could not find {input_path}")
        return

    print("============================================================")
    print("Initiating 2-Step SkyPortal (Fritz) Push Sequence")
    print(f"Target Group ID: {args.group_id}")
    print("============================================================")

    # 1. Read JSONL verdicts
    verdicts = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            verdicts.append(json.loads(line))
            
    print(f"Loaded {len(verdicts)} anomalies perfectly primed for Active Learning.")
    print("Beginning UI pipeline inference insertion...\n")

    # 2. Sequential 2-Step Push
    success_count = 0
    for idx, verdict in enumerate(verdicts):
        oid = verdict.get("oid")
        if not oid: continue

        print(f"[{idx+1}/{len(verdicts)}] Processing {oid} ({verdict.get('primary_class', 'Unknown')})...")
        survey = "ZTF" if oid.startswith("ZTF") else "LSST"
        
        # Step 1: Force Fritz backend to construct the lightcurve dashboard
        pushed_obj = push_anomaly_to_fritz(oid, survey, args.group_id, args.token)
        
        if pushed_obj:
            # Step 2: Push the AI annotations into the schema for the UI widget to render
            push_annotation_to_fritz(oid, verdict, args.group_id, args.token)
            success_count += 1
            
        time.sleep(args.delay) # Rate limit respect

    print("\n============================================================")
    print("DELIVERY SUMMARY")
    print(f"Successfully processed {success_count}/{len(verdicts)} anomalies.")
    print("The SkyPortal CYOA Frontend is now populated.")
    print("============================================================")

if __name__ == "__main__":
    main()
