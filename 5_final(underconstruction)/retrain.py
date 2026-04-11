"""
retrain.py — 4-Source Self-Improvement Loop for BOOM Anomaly Layer.

Collects training labels from 4 sources with a trust hierarchy:
  1. Human feedback (weight=1.0)
  2. LLM verdicts (weight=0.7 × confidence)
  3. Self-validation against XGBoost (weight=0.5)
  4. Pseudo-labels from high-confidence scores (weight=0.3)

Retrains per-profile GradientBoostingClassifier and shared Isolation Forest.

Adapted from anomaly_hunter/backend/retrain.py — architecturally identical,
only input dimensionality changes from 64 to 124.
"""
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .config import (
    FEATURE_DIM,
    MIN_LABELS_FOR_RETRAIN,
    FEEDBACK_WEIGHTS,
    PSEUDO_POSITIVE_THRESHOLD, PSEUDO_NEGATIVE_THRESHOLD,
    GB_N_ESTIMATORS, GB_MAX_DEPTH, GB_LEARNING_RATE, GB_SUBSAMPLE,
)


def collect_profile_training_data(storage, feature_matrix, source_ids, profile_key):
    """
    Collect training data from all 4 sources, scoped to a specific profile.

    Args:
        storage: AnomalyStorage instance
        feature_matrix: (N, 124) feature matrix
        source_ids: list of source IDs (same order as feature_matrix)
        profile_key: profile identifier

    Returns:
        (embeddings, labels, weights, sources) — all numpy arrays
    """
    # Build source_id → feature_vector mapping
    id_to_idx = {sid: i for i, sid in enumerate(source_ids)}

    all_emb, all_labels, all_weights, all_sources = [], [], [], []

    # ── Source 1: Human feedback ──────────────────────────────────────
    feedback_df = storage.load_feedback(profile_key)
    n_human = 0

    if not feedback_df.empty:
        for _, row in feedback_df.iterrows():
            sid = row["source_id"]
            if sid not in id_to_idx:
                continue
            idx = id_to_idx[sid]
            action = row["action"]

            if action == "interesting":
                all_emb.append(feature_matrix[idx])
                all_labels.append(1.0)
                all_weights.append(FEEDBACK_WEIGHTS["human"])
                all_sources.append("human")
                n_human += 1
            elif action == "noise":
                all_emb.append(feature_matrix[idx])
                all_labels.append(0.0)
                all_weights.append(FEEDBACK_WEIGHTS["human"])
                all_sources.append("human")
                n_human += 1
            elif action == "classify":
                all_emb.append(feature_matrix[idx])
                all_labels.append(0.3)
                all_weights.append(0.7)
                all_sources.append("human")
                n_human += 1

    print(f"[Retrain/{profile_key}] Source 1 (human): {n_human} labels")

    # ── Source 2: Self-validation against XGBoost ─────────────────────
    scores_df = storage.load_scores(profile_key)
    n_selfval = 0

    if not scores_df.empty:
        # Objects that were classified with high XGBoost confidence
        classified = scores_df[
            (scores_df["triage"] == "classified") & (scores_df["xgb_confidence"] > 0.90)
        ]
        for _, row in classified.iterrows():
            sid = row["source_id"]
            if sid not in id_to_idx:
                continue
            idx = id_to_idx[sid]
            all_emb.append(feature_matrix[idx])
            all_labels.append(0.2)
            all_weights.append(FEEDBACK_WEIGHTS["self_val"])
            all_sources.append("self_val")
            n_selfval += 1

    print(f"[Retrain/{profile_key}] Source 2 (self-validation): {n_selfval} labels")

    # ── Source 3: Pseudo-labels from high-confidence scores ───────────
    n_pseudo = 0
    if not scores_df.empty:
        for _, row in scores_df.iterrows():
            sid = row["source_id"]
            if sid not in id_to_idx:
                continue
            idx = id_to_idx[sid]
            score = row.get("anomaly_score", 0.5)

            if score >= PSEUDO_POSITIVE_THRESHOLD:
                all_emb.append(feature_matrix[idx])
                all_labels.append(1.0)
                all_weights.append(FEEDBACK_WEIGHTS["pseudo"])
                all_sources.append("pseudo")
                n_pseudo += 1
            elif score <= PSEUDO_NEGATIVE_THRESHOLD:
                all_emb.append(feature_matrix[idx])
                all_labels.append(0.0)
                all_weights.append(FEEDBACK_WEIGHTS["pseudo"])
                all_sources.append("pseudo")
                n_pseudo += 1

    print(f"[Retrain/{profile_key}] Source 3 (pseudo-labels): {n_pseudo} labels")

    # ── Source 4: LLM verdicts from this profile ──────────────────────
    reviews_df = storage.load_reviews(profile_key)
    n_llm = 0

    if not reviews_df.empty:
        for _, row in reviews_df.iterrows():
            sid = row["source_id"]
            if sid not in id_to_idx:
                continue
            idx = id_to_idx[sid]
            verdict = row["verdict"]
            confidence = float(row.get("confidence", 0.5))

            if verdict == "interesting":
                label = 1.0
            elif verdict == "noise":
                label = 0.0
            elif verdict == "known_type":
                label = 0.2
            else:
                continue

            all_emb.append(feature_matrix[idx])
            all_labels.append(label)
            all_weights.append(FEEDBACK_WEIGHTS["llm"] * confidence)
            all_sources.append("llm")
            n_llm += 1

    print(f"[Retrain/{profile_key}] Source 4 (LLM): {n_llm} labels")
    total = len(all_emb)
    print(f"[Retrain/{profile_key}] Total: {total} training samples")

    if not all_emb:
        return np.array([]), np.array([]), np.array([]), []

    return (
        np.array(all_emb, dtype=np.float32),
        np.array(all_labels, dtype=np.float32),
        np.array(all_weights, dtype=np.float32),
        all_sources,
    )


def train_profile_classifier(profile, embeddings, labels, weights):
    """
    Train this profile's feedback classifier (GradientBoostingClassifier).

    Args:
        profile: BoomScienceProfile instance
        embeddings: (N, 124) array
        labels: (N,) array
        weights: (N,) sample weights

    Returns:
        True if training succeeded
    """
    if len(embeddings) < MIN_LABELS_FOR_RETRAIN:
        print(f"[Retrain/{profile.key}] ⚠️ Only {len(embeddings)} labels, "
              f"need ≥{MIN_LABELS_FOR_RETRAIN}")
        return False

    # Binary labels (threshold at 0.5)
    y_binary = (labels >= 0.5).astype(int)

    if len(np.unique(y_binary)) < 2:
        print(f"[Retrain/{profile.key}] ⚠️ Need both positive and negative examples")
        return False

    print(f"[Retrain/{profile.key}] 🧠 Training feedback classifier on "
          f"{len(embeddings)} samples...")
    print(f"   Positives: {y_binary.sum()} | Negatives: {(1 - y_binary).sum()}")

    clf = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        max_depth=GB_MAX_DEPTH,
        learning_rate=GB_LEARNING_RATE,
        subsample=GB_SUBSAMPLE,
        random_state=42,
    )
    clf.fit(embeddings, y_binary, sample_weight=weights)

    profile.feedback_clf = clf
    profile.feedback_clf_trained = True
    profile.save_feedback_clf()

    print(f"[Retrain/{profile.key}] ✅ Feedback classifier trained and saved")
    return True


def run_profile_retraining(profile, storage, feature_matrix, source_ids):
    """
    Full retraining pipeline for a single profile.

    1. Collect labels from 4 sources (scoped to this profile)
    2. Train this profile's feedback classifier
    """
    print(f"\n{'─' * 40}")
    print(f"  🧠 RETRAINING: {profile.name}")
    print(f"{'─' * 40}")

    emb, labels, weights, sources = collect_profile_training_data(
        storage, feature_matrix, source_ids, profile.key
    )

    if len(emb) >= MIN_LABELS_FOR_RETRAIN:
        train_profile_classifier(profile, emb, labels, weights)
    else:
        print(f"[Retrain/{profile.key}] ⏳ Not enough labels yet "
              f"({len(emb)}/{MIN_LABELS_FOR_RETRAIN})")

    print(f"{'─' * 40}\n")
