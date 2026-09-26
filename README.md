# Amazon ML Challenge 2026 — Business Entity Resolution (Team BatSignal)

## Experiment: Class Weighting vs. Explicit Hard-Negative Mining

**Author**: Punith (`punithsai2006`)  
**Team**: BatSignal  
**Task**: Business Entity Resolution (Record Linkage)

---

## 📌 Executive Summary & Key Results

This repository contains the complete experimental evaluation comparing **Class Weighting** (`scale_pos_weight`) against **Explicit Hard-Negative Mining** on the Amazon ML Challenge 2026 Business Entity Resolution task.

### Results Summary Table

| Experiment Strategy | Training Candidate Pairs | Positive Matches | Negative Candidates | Imbalance Ratio (N:P) | Optimal Threshold | Validation Macro $F_{0.5}$ Score | Status |
|---|---|---|---|---|---|---|---|
| **Control Baseline** | 75,000 | 7,129 | 67,871 | 9.52 : 1 | **0.3800** | **0.620746** | Verified |
| **Strategy A: Class Weighting** | 75,000 | 7,129 | 67,871 | 9.52 : 1 | **0.8400** | **0.620184** | Verified |
| **Strategy B: Hard-Negative Mining** | 42,774 | 7,129 | 35,645 | 5.00 : 1 | **0.9000** | **0.206176** | Verified (Degraded) |

### Key Architectural Conclusion for Team BatSignal
1. **Do NOT use explicit hard-negative sampling** (dropping "easy" negative candidate pairs during training). Dropping easy negatives prevents the classifier from establishing a proper low-similarity decision boundary, causing a **15x increase in False Positives** during full candidate set inference ($F_{0.5}$ drops from `0.6207` to `0.2062`).
2. **Class Weighting + Post-Hoc Threshold Optimization** (Strategy A) preserves full candidate coverage while achieving optimal $F_{0.5}$ performance.

---

## 📂 Repository Structure

```
.
├── code/
│   └── business_entity_resolution/
│       └── src/
│           ├── normalization.py         # Text cleaning & standardization
│           ├── blocking.py              # Multi-stage TF-IDF candidate pair blocker
│           ├── features.py              # Pairwise string & structural feature extraction
│           ├── train.py                 # HistGradientBoosting classifier training logic
│           ├── infer.py                 # High-throughput candidate prediction & thresholding
│           ├── metric.py                # Official Amazon ML Macro F0.5 evaluation metric
│           ├── verify_metric.py         # Verification script matching worked examples (100% pass)
│           └── run_experiments.py       # End-to-end parallel experiment execution engine
├── splits/
│   └── val_entity_ids.txt              # Strict 20% entity-level validation split (441,364 S1 IDs)
├── experiments/
│   ├── results_summary.csv             # Raw CSV metric comparison
│   ├── class_ratio_report.md           # Imbalance ratio measurement report (9.52:1)
│   ├── experiment_class_weight.md      # Strategy A execution report
│   ├── experiment_hard_negative.md     # Strategy B execution report
│   └── error_analysis.md               # Detailed error breakdown (FP, FN, Singleton FP)
├── EXPERIMENT_REPORT_HARD_NEGATIVES_VS_CLASS_WEIGHTS.md # Full technical report
└── README.md                            # Comprehensive team usage guide
```

---

## 🚀 Quick Start & How Team Members Can Use This Work

### 1. Requirements & Prerequisites
Ensure Python 3.10+ and standard ML libraries are installed:
```bash
pip install pandas numpy scikit-learn
```

### 2. Verify Evaluation Metric
Run the metric verification script to confirm exact compliance with the official Amazon ML worked example ($P=2/3, R=1.0 \Rightarrow F_{0.5}=0.714286$):
```bash
python code/business_entity_resolution/src/verify_metric.py
```

### 3. Re-run All Experiments
To execute all experiments (Control, Strategy A, and Strategy B) with parallel multi-core processing:
```bash
python code/business_entity_resolution/src/run_experiments.py
```
All outputs and reports will be updated automatically in the `experiments/` directory.

### 4. How to Train & Infer on New Data
```python
from code.business_entity_resolution.src.blocking import MultiStageBlocker
from code.business_entity_resolution.src.features import build_features_ultra_fast
from code.business_entity_resolution.src.train import train_classifier
from code.business_entity_resolution.src.infer import predict_entity_matches

# 1. Generate Candidates
blocker = MultiStageBlocker(top_k=15)
candidates = blocker.generate_candidates_for_source(s1_df, s2_df, "S2")

# 2. Build Pairwise Features
feature_df = build_features_ultra_fast(candidates, s1_df, s2_df)

# 3. Train Classifier
model = train_classifier(feature_df[FEATURE_COLS], feature_df["label"], class_weight="balanced")

# 4. Predict Matches with Post-Hoc Optimal Threshold (0.84)
predictions = predict_entity_matches(model, feature_df, threshold=0.84)
```

---

## 🔍 Detailed Error Analysis Summary

From [experiments/error_analysis.md](experiments/error_analysis.md):

1. **False Positives (151 pairs)**: Common business names sharing high string similarity but situated at completely different addresses.
2. **False Negatives (3,269 pairs)**: Uncaptured legal suffix variations ("Private Limited" vs "Pvt Ltd") and minor typos in street addresses.
3. **Singleton False Positives (7 entities)**: True singleton entities with no actual matches where weak candidates barely crossed threshold.

---

## 📜 Citation & Attribution
- **Author**: Punith (`punithsai2006`)
- **Challenge**: Amazon ML Challenge 2026
- **Team**: BatSignal
