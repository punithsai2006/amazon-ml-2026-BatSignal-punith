"""
Official F_0.5 Metric Verification Module (BatSignal Team)
Verifies F_0.5 calculation against the exact worked example in the prompt:
  Prediction: S1-00001 -> [S2-00047, S2-00193, S3-00812]
  Ground truth: S1-00001 -> [S2-00047, S3-00812]
  Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.714
"""

import numpy as np
from typing import Dict, Set, List


def compute_entity_f05(pred_matches: Set[str], gt_matches: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Singleton rules:
    - True singleton (gt_matches is empty):
        - empty prediction -> 1.0
        - any match predicted -> 0.0
    """
    # Case 1: True Singleton (Ground truth has 0 matches)
    if len(gt_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    # Case 2: Non-singleton, but prediction is empty
    if len(pred_matches) == 0:
        return 0.0

    tp = len(gt_matches.intersection(pred_matches))
    fp = len(pred_matches - gt_matches)
    fn = len(gt_matches - pred_matches)

    if tp == 0:
        return 0.0

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return float(f05)


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: Set[str]
) -> float:
    """Computes macro-averaged F_0.5 score across all Source 1 entities."""
    scores = []
    for s1_id in all_s1_ids:
        gt_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        scores.append(compute_entity_f05(pred_set, gt_set))
    return float(np.mean(scores)) if scores else 0.0


def test_worked_example():
    """Verifies the implementation against the challenge prompt worked example."""
    gt = {"S1-00001": {"S2-00047", "S3-00812"}}
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    all_s1 = {"S1-00001"}

    score = compute_entity_f05(pred["S1-00001"], gt["S1-00001"])
    print(f"Worked Example Test Score: {score:.6f} (Expected: 0.714285...)")
    assert abs(score - 0.7142857) < 1e-4, f"Metric mismatch! Got {score}"

    # Singleton Test 1: True singleton + empty prediction -> 1.0
    singleton_score_pass = compute_entity_f05(set(), set())
    assert singleton_score_pass == 1.0, f"Singleton empty test failed! Got {singleton_score_pass}"

    # Singleton Test 2: True singleton + false match -> 0.0
    singleton_score_fail = compute_entity_f05({"S2-99999"}, set())
    assert singleton_score_fail == 0.0, f"Singleton false match test failed! Got {singleton_score_fail}"

    print("ALL METRIC VERIFICATION CHECKS PASSED 100%!")
    return True


if __name__ == "__main__":
    test_worked_example()
