"""
Experiment Module: Class-Weighting vs. Hard-Negative Mining (BatSignal Team)
Task Assignment:
1. Measures positive/negative class ratio in candidate pairs from blocking.
2. Evaluates Strategy (a) Class-Weighting (scale_pos_weight) vs. Strategy (b) Explicit Hard-Negative Mining
   on the exact validation split in `splits/val_entity_ids.txt`.
3. Performs error analysis (FP, FN, Singleton FP) to guide feature additions.
4. Reports exact validation F_0.5 score for all experiment variants.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any

# Ensure src module is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.ensemble import HistGradientBoostingClassifier
from verify_metric import compute_entity_f05, compute_macro_f05

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_validation_entity_ids(val_split_path: str = "splits/val_entity_ids.txt") -> Set[str]:
    """Loads validation S1 entity IDs from the team's shared validation split file."""
    if not os.path.exists(val_split_path):
        logger.warning(f"Validation split file not found at {val_split_path}. Returning empty set.")
        return set()
    with open(val_split_path, "r", encoding="utf-8") as f:
        val_ids = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    logger.info(f"Loaded {len(val_ids)} validation entity IDs from {val_split_path}")
    return val_ids


def measure_class_imbalance(
    candidates_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Measures and reports the positive/negative class ratio in candidate pairs.
    Candidate pairs schema: source1_entity_id (s1_id), candidate_id (matched_record_id).
    Ground truth schema: s1_id, matched_record_id.
    """
    if candidates_df.empty or ground_truth_df.empty:
        logger.warning("Candidates or Ground Truth DataFrame is empty. Skipping class ratio measurement.")
        return {"positives": 0, "negatives": 0, "ratio": 0.0}

    s1_col = "s1_id" if "s1_id" in candidates_df.columns else "source1_entity_id"
    cand_col = "candidate_id" if "candidate_id" in candidates_df.columns else "candidate_entity_ids"
    gt_s1_col = "s1_id" if "s1_id" in ground_truth_df.columns else "source1_entity_id"
    gt_cand_col = "matched_record_id" if "matched_record_id" in ground_truth_df.columns else "matched_entity_ids"

    gt_pairs = set(zip(ground_truth_df[gt_s1_col], ground_truth_df[gt_cand_col]))
    cand_pairs = list(zip(candidates_df[s1_col], candidates_df[cand_col]))

    pos_count = sum(1 for p in cand_pairs if p in gt_pairs)
    neg_count = len(cand_pairs) - pos_count
    ratio = neg_count / float(pos_count) if pos_count > 0 else 0.0

    report = {
        "total_candidates": len(cand_pairs),
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "negative_to_positive_ratio": ratio
    }

    logger.info(f"--- CANDIDATE BLOCKING CLASS RATIO ---")
    logger.info(f"Total Candidate Pairs : {report['total_candidates']}")
    logger.info(f"Positive Matches      : {report['positive_pairs']}")
    logger.info(f"Negative Candidates   : {report['negative_pairs']}")
    logger.info(f"Imbalance Ratio (N:P) : {report['negative_to_positive_ratio']:.2f} : 1")

    return report


def run_experiment_comparison(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    val_s1_ids: Set[str],
    feature_cols: List[str]
) -> Dict[str, float]:
    """
    Compares Strategy (a) Class-Weighting vs. Strategy (b) Hard-Negative Mining
    on the exact validation split.
    """
    logger.info("\n=======================================================")
    logger.info("EXPERIMENT: Class-Weighting vs. Hard-Negative Mining")
    logger.info("=======================================================")

    results = {}

    # Strategy (a): Class-Weighting (Using scale_pos_weight or class sample weighting)
    logger.info("\nEvaluating Strategy (a): Class-Weighting (scale_pos_weight)...")
    clf_weighted = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    X_tr = train_features[feature_cols]
    y_tr = train_features["label"]
    clf_weighted.fit(X_tr, y_tr)

    # Validate on exact val_s1_ids split
    X_val = val_features[feature_cols]
    val_probs = clf_weighted.predict_proba(X_val)[:, 1]
    val_scored = val_features.copy()
    val_scored["probability"] = val_probs

    # Evaluate validation F_0.5 at threshold = 0.50
    val_preds_a = val_scored[val_scored["probability"] >= 0.50]
    pred_dict_a = {}
    for _, r in val_preds_a.iterrows():
        s1 = r.get("s1_id", r.get("source1_entity_id"))
        cand = r.get("candidate_id", r.get("candidate_entity_ids"))
        if s1 not in pred_dict_a:
            pred_dict_a[s1] = set()
        pred_dict_a[s1].add(cand)

    gt_dict = {}
    for _, r in ground_truth_df.iterrows():
        s1 = r.get("s1_id", r.get("source1_entity_id"))
        cand = r.get("matched_record_id", r.get("matched_entity_ids"))
        if s1 not in gt_dict:
            gt_dict[s1] = set()
        gt_dict[s1].add(cand)

    score_a = compute_macro_f05(pred_dict_a, gt_dict, val_s1_ids)
    results["strategy_a_class_weighting_f05"] = score_a
    logger.info(f"Strategy (a) Class-Weighting Validation F_0.5 Score: {score_a:.6f}")

    # Strategy (b): Hard-Negative Mining
    logger.info("\nEvaluating Strategy (b): Hard-Negative Mining...")
    positives = train_features[train_features["label"] == 1]
    negatives = train_features[train_features["label"] == 0]

    # Select hard negatives sorted by top similarity score
    hard_neg_count = min(len(negatives), len(positives) * 5)
    sort_col = "composite_sim" if "composite_sim" in negatives.columns else feature_cols[0]
    hard_negatives = negatives.sort_values(by=sort_col, ascending=False).head(hard_neg_count)
    hard_neg_train = pd.concat([positives, hard_negatives], ignore_index=True).sample(frac=1.0, random_state=42)

    clf_hard_neg = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    clf_hard_neg.fit(hard_neg_train[feature_cols], hard_neg_train["label"])

    val_probs_b = clf_hard_neg.predict_proba(X_val)[:, 1]
    val_scored_b = val_features.copy()
    val_scored_b["probability"] = val_probs_b

    val_preds_b = val_scored_b[val_scored_b["probability"] >= 0.50]
    pred_dict_b = {}
    for _, r in val_preds_b.iterrows():
        s1 = r.get("s1_id", r.get("source1_entity_id"))
        cand = r.get("candidate_id", r.get("candidate_entity_ids"))
        if s1 not in pred_dict_b:
            pred_dict_b[s1] = set()
        pred_dict_b[s1].add(cand)

    score_b = compute_macro_f05(pred_dict_b, gt_dict, val_s1_ids)
    results["strategy_b_hard_negatives_f05"] = score_b
    logger.info(f"Strategy (b) Hard-Negative Mining Validation F_0.5 Score: {score_b:.6f}")

    logger.info("\n--- EXPERIMENT SUMMARY REPORT ---")
    logger.info(f"Strategy (a) Class-Weighting F_0.5 : {score_a:.6f}")
    logger.info(f"Strategy (b) Hard-Negatives F_0.5  : {score_b:.6f}")
    
    if score_b > score_a:
        logger.info(f"WINNER: Strategy (b) Hard-Negative Mining (+{(score_b - score_a):.6f} F_0.5 gain)")
    else:
        logger.info(f"WINNER: Strategy (a) Class-Weighting (+{(score_a - score_b):.6f} F_0.5 gain)")

    return results


if __name__ == "__main__":
    val_ids = load_validation_entity_ids()
    print("Validation Entity Count:", len(val_ids))
