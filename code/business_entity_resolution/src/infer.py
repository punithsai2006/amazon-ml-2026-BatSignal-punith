"""
Inference & Prediction Scoring Module (BatSignal Team Baseline)
Loads trained model, scores candidate pair feature vectors,
and applies optimal decision probability threshold.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any

try:
    from train import FEATURE_COLS
except ImportError:
    from src.train import FEATURE_COLS

logger = logging.getLogger(__name__)


def score_candidate_pairs(model: Any, feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Predicts match probabilities for candidate pair feature vectors."""
    if feature_matrix.empty:
        df = feature_matrix.copy()
        df["probability"] = 0.0
        return df

    X = feature_matrix[FEATURE_COLS]
    probabilities = model.predict_proba(X)[:, 1]

    result_df = feature_matrix.copy()
    result_df["probability"] = probabilities
    return result_df


def predict_matches(
    scored_df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Filters candidate pairs exceeding the decision threshold."""
    if scored_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

    s1_col = "s1_id" if "s1_id" in scored_df.columns else "source1_entity_id"
    cand_col = "candidate_id" if "candidate_id" in scored_df.columns else "candidate_entity_ids"

    matches = scored_df[scored_df["probability"] >= threshold].copy()

    # Format output as grouped per Source 1 entity: source1_entity_id \t matched_entity_ids
    grouped = matches.groupby(s1_col)[cand_col].apply(lambda ids: ",".join(list(ids))).reset_index()
    grouped.columns = ["source1_entity_id", "matched_entity_ids"]

    logger.info(f"Predicted {len(matches)} matches out of {len(scored_df)} candidates at threshold {threshold:.4f}")
    return grouped
