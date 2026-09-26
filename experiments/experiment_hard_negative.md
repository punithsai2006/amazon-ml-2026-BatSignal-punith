# Experiment Report B: Explicit Hard-Negative Mining

**Team**: BatSignal  
**Date**: 2026-09-26 17:35:36  
**Validation Split**: `splits/val_entity_ids.txt`  
**Status**: Executed & Verified

---

## Hyperparameters & Strategy
- **Sampling Strategy**: Explicit Hard-Negative Mining (Top 5:1 hard negatives sorted by `composite_sim`)
- **Classifier**: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6)`
- **Training Positives**: 7,129
- **Hard Negatives Sampled**: 35,645
- **Sampling Ratio**: 5.0 : 1

---

## Validation Performance
- **Optimal Threshold**: `0.9000`
- **Validation Macro $F_0.5$ Score**: **`0.206176`**
- **Execution Runtime**: `78.05 seconds`
