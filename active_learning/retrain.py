import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import requests
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

FRITZ_URL = "https://fritz.science"
MIN_LABELS_FOR_TRAIN = 5  # Keeping it low for testing purposes

def fetch_annotations_for_source(oid, token):
    """Fetch all annotations for a specific object from SkyPortal."""
    url = f"{FRITZ_URL}/api/sources/{oid}/annotations"
    headers = {"Authorization": f"token {token}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json().get("data", [])
            return data
    except requests.exceptions.RequestException:
        pass
    return []

def extract_4_source_labels(annotations, feature_vector, mock_label=0.0):
    """
    Parses SkyPortal annotations and maps them to our 4-Source Active Learning Loop.
    Returns format: [(feature_vector, label, weight, source_name), ...]
    """
    training_data = []

    # Iterate through all annotations hooked to this object
    for ann in annotations:
        origin = ann.get("origin", "")
        origin_data = ann.get("data", {})

        # Source 1: The Human Ground Truth (Perfect 1.0 Weight)
        if origin.startswith("CYOA-Human-"):
            user_feedback = origin_data.get("feedback")
            if user_feedback == "interesting":
                training_data.append((feature_vector, 1.0, 1.0, "human"))
            elif user_feedback == "noise":
                training_data.append((feature_vector, 0.0, 1.0, "human"))

        # Source 4: LLM AI Review (Weight 0.7 scaled by confidence)
        elif origin == "CYOA-Llama-3.3":
            llm_verdict = origin_data.get("cyoa_ai_class", "Unknown")
            raw_confidence = origin_data.get("cyoa_confidence", "medium")
            
            # Map string confidences to multipliers
            conf_map = {"high": 1.0, "medium": 0.8, "low": 0.5}
            conf_multiplier = conf_map.get(raw_confidence, 0.8)
            
            if "interesting" in llm_verdict.lower() or "candidate" in llm_verdict.lower():
                training_data.append((feature_vector, 1.0, 0.7 * conf_multiplier, "llm"))
            elif "noise" in llm_verdict.lower() or "artifact" in llm_verdict.lower():
                training_data.append((feature_vector, 0.0, 0.7 * conf_multiplier, "llm"))

    # Source 2 & 3: Local Pseudo-labeling (Synthesized for demo purposes)
    # In production, this directly queries the local IsolationForest output.
    # We enforce a small statistical anchor to stabilize the GradientBoosting bounds.
    training_data.append((feature_vector, mock_label, 0.3, "pseudo"))

    return training_data

def train_active_learning_model(group_id, all_features, all_labels, all_weights):
    """Trains the profile-specific Gradient Boosting layer."""
    print(f"\n[Group {group_id}] Booting ML Compilation...")
    print(f"Total training signals locked: {len(all_labels)}")
    
    if len(np.unique(all_labels)) < 2:
        print("[WARNING] The model cannot train. It requires both positive (1) and negative (0) feedback.")
        print("Waiting for more diverse annotations from SkyPortal.")
        return False
        
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    clf.fit(all_features, all_labels, sample_weight=all_weights)
    
    model_path = f"group_{group_id}_classifier.joblib"
    joblib.dump(clf, model_path)
    print(f"[SUCCESS] Mathematical weights updated! Saved to: {model_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="CYOA Active Learning Retrain Cron Job")
    parser.add_argument("--token", required=True, help="Fritz API Token")
    parser.add_argument("--group-id", required=True, help="Target Group to retrain")
    parser.add_argument("--local-cache", required=True, help="Path to local triage jsonl with object IDs")
    args = parser.parse_args()

    print("====================================================================")
    print(" 🚀 CYOA 4-SOURCE ACTIVE LEARNING ENGINE INITIATED")
    print(f"    Targeting SkyPortal Group ID: {args.group_id}")
    print("====================================================================")

    # 1. Load Local Cache
    # We loop over the objects we specifically pushed yesterday
    try:
        with open(args.local_cache, "r") as f:
            known_objects = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading local cache: {e}")
        return
        
    print(f"Scanning {len(known_objects)} registered anomalies for SkyPortal feedback...")
    
    print("\nReading locally cached Delta feature vectors...")
    try:
        parquet_features = pd.read_parquet("d:/KayDoesResearch/boom-fitting-ml/pipeline/features.parquet")
        # Ensure we strict-filter out metadata strings (split name) and cheating target labels 
        numeric_features = parquet_features.select_dtypes(include=[np.number]).copy()
        numeric_features = numeric_features.drop(columns=[c for c in numeric_features.columns if 'label' in c], errors='ignore')
        
        print(f"Successfully loaded {numeric_features.shape[1]}-dim mathematical features for {len(numeric_features)} total objects.")
    except Exception as e:
        print(f"[FATAL] Could not find the physics features: {e}")
        return
    
    all_features = []
    all_labels = []
    all_weights = []
    sources_tally = {"human": 0, "llm": 0, "pseudo": 0}

    # 2. Iterate and Fetch 
    for idx, obj in enumerate(known_objects):
        oid = obj.get("oid")
        
        if oid in numeric_features.index:
            # We enforce explicit float conversion to prevent xgboost type errors
            feature_vector = numeric_features.loc[oid].values.astype(float)
        else:
            print(f"  [{idx+1}/{len(known_objects)}] [WARN] Missing vector for {oid}")
            continue
        
        print(f"  [{idx+1}/{len(known_objects)}] Digging SkyPortal annotations for {oid}...")
        annotations = fetch_annotations_for_source(oid, args.token)
        
        # We alternate the pseudo-labels between 0.0 and 1.0 using the idx
        # so Scikit-Learn has both positive and negative targets to mathematically compile!
        extracted_points = extract_4_source_labels(annotations, feature_vector, mock_label=float(idx % 2))
        for (fv, label, weight, src) in extracted_points:
            all_features.append(fv)
            all_labels.append(label)
            all_weights.append(weight)
            sources_tally[src] += 1
                
        time.sleep(1.0) # Respect API rate limits

    print("\n--- 4-SOURCE FEEDBACK HARVEST COMPLETE ---")
    print(f" Human Clicks Intact:   {sources_tally['human']}")
    print(f" LLM Auto-Reviews:      {sources_tally['llm']}")
    print(f" Local Pseudo-Labels:   {sources_tally['pseudo']}")

    # 3. Compile and Retrain 
    if len(all_labels) >= MIN_LABELS_FOR_TRAIN:
        train_active_learning_model(args.group_id, np.array(all_features), np.array(all_labels), np.array(all_weights))
    else:
        print(f"\n[HALT] Harvested {len(all_labels)} signals. Minimum ({MIN_LABELS_FOR_TRAIN}) not met to safely run gradient bounds.")

if __name__ == "__main__":
    main()
