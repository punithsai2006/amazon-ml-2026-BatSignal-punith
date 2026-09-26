"""
Official F_0.5 Metric & Threshold Tuning Module (BatSignal Team Baseline)
Implements official per-entity F_0.5 metric with singleton handling rules
and grid-search threshold optimization.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple

try:
    from verify_metric import compute_entity_f05, compute_macro_f05
except ImportError:
    from src.verify_metric import compute_entity_f05, compute_macro_f05

logger = logging.getLogger(__name__)


def find_optimal_threshold(
    val_scored_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    all_s1_ids: Set[str],
    search_min: float = 0.1,
    search_max: float = 0.9,
    step: float = 0.02
) -> Tuple[float, float]:
    """
    Grid-searches decision probability threshold to maximize validation macro F_0.5.
    Returns (best_threshold, best_f05_score).
    """
    if val_scored_df.empty:
        return 0.5, 0.0

    s1_col = "s1_id" if "s1_id" in val_scored_df.columns else "source1_entity_id"
    cand_col = "candidate_id" if "candidate_id" in val_scored_df.columns else "candidate_entity_ids"

    gt_s1_col = "s1_id" if "s1_id" in ground_truth_df.columns else "source1_entity_id"
    gt_cand_col = "matched_record_id" if "matched_record_id" in ground_truth_df.columns else "matched_entity_ids"

    gt_dict: Dict[str, Set[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1 = row[gt_s1_col]
        raw_cands = str(row[gt_cand_col])
        if s1 not in gt_dict:
            gt_dict[s1] = set()
        for cand in raw_cands.split(","):
            cand = cand.strip()
            if cand:
                gt_dict[s1].add(cand)

    best_thresh = 0.5
    best_score = -1.0

    thresholds = np.arange(search_min, search_max + step, step)
    for thresh in thresholds:
        matches_df = val_scored_df[val_scored_df["probability"] >= thresh]
        pred_dict: Dict[str, Set[str]] = {}
        for _, row in matches_df.iterrows():
            s1 = row[s1_col]
            cand = row[cand_col]
            if s1 not in pred_dict:
                pred_dict[s1] = set()
            pred_dict[s1].add(cand)

        score = compute_macro_f05(pred_dict, gt_dict, all_s1_ids)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    logger.info(f"Optimal threshold found: {best_thresh:.4f} with validation F_0.5 score: {best_score:.4f}")
    return best_thresh, best_score
