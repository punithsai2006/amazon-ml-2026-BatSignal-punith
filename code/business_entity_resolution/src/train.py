"""
Pairwise Match Classifier Training Module (BatSignal Team Baseline)
Trains Gradient Boosted Trees (HistGradientBoostingClassifier / XGBoost)
"""

import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import HistGradientBoostingClassifier

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "name_levenshtein", "name_jaccard", "name_token_sort", "name_ngram_jaccard",
    "name_length_diff", "name_length_ratio", "addr_levenshtein", "addr_jaccard",
    "addr_ngram_jaccard", "addr_length_ratio", "country_match", "country_missing",
    "composite_sim"
]


def prepare_training_labels(
    feature_matrix: pd.DataFrame,
    ground_truth_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attaches ground truth binary match labels (1 for true match, 0 for negative candidate).
    Ground truth schema: s1_id / source1_entity_id, matched_record_id / matched_entity_ids.
    """
    if feature_matrix.empty or ground_truth_df.empty:
        df = feature_matrix.copy()
        df["label"] = 0
        return df

    s1_col = "s1_id" if "s1_id" in feature_matrix.columns else "source1_entity_id"
    cand_col = "candidate_id" if "candidate_id" in feature_matrix.columns else "candidate_entity_ids"

    gt_s1_col = "s1_id" if "s1_id" in ground_truth_df.columns else "source1_entity_id"
    gt_cand_col = "matched_record_id" if "matched_record_id" in ground_truth_df.columns else "matched_entity_ids"

    # Expand GT matched_entity_ids if comma-separated
    gt_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1 = row[gt_s1_col]
        raw_cands = str(row[gt_cand_col])
        for cand in raw_cands.split(","):
            cand = cand.strip()
            if cand:
                gt_pairs.add((s1, cand))

    def is_match(row):
        return 1 if (row[s1_col], row[cand_col]) in gt_pairs else 0

    df = feature_matrix.copy()
    df["label"] = df.apply(is_match, axis=1)

    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    logger.info(f"Training dataset labeled: {pos_count} positives, {neg_count} negatives.")
    return df


def select_hard_negatives(
    labeled_df: pd.DataFrame,
    neg_to_pos_ratio: float = 5.0
) -> pd.DataFrame:
    """
    Hard negative mining: prioritizes high-similarity non-matching candidate pairs.
    """
    positives = labeled_df[labeled_df["label"] == 1]
    negatives = labeled_df[labeled_df["label"] == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return labeled_df

    target_neg_count = int(len(positives) * neg_to_pos_ratio)
    if len(negatives) <= target_neg_count:
        return labeled_df

    # Sort negatives by composite similarity descending to select hard negatives
    sort_col = "composite_sim" if "composite_sim" in negatives.columns else FEATURE_COLS[0]
    hard_negatives = negatives.sort_values(by=sort_col, ascending=False).head(target_neg_count)
    sampled_df = pd.concat([positives, hard_negatives], ignore_index=True).sample(frac=1.0, random_state=42)

    logger.info(f"Hard negative sampling selected {len(hard_negatives)} hard negatives for {len(positives)} positives.")
    return sampled_df


def train_classifier(
    training_data: pd.DataFrame,
    class_weight: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None
) -> HistGradientBoostingClassifier:
    """Trains Gradient Boosting pairwise match classifier."""
    X = training_data[FEATURE_COLS]
    y = training_data["label"]

    params = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42
    }
    if class_weight:
        params["class_weight"] = class_weight
    if model_params:
        params.update(model_params)

    clf = HistGradientBoostingClassifier(**params)
    clf.fit(X, y)
    logger.info("Pairwise match classifier trained successfully.")
    return clf


def save_model(model: Any, filepath: str = "models/xgboost_entity_matching.pkl") -> None:
    """Saves serialized model artifact."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model artifact saved to {filepath}")


def load_model(filepath: str = "models/xgboost_entity_matching.pkl") -> Any:
    """Loads serialized model artifact."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
