# Experiment Report A: Class Weighting (`scale_pos_weight`)

**Team**: BatSignal  
**Date**: 2026-09-26 17:35:36  
**Validation Split**: `splits/val_entity_ids.txt`  
**Status**: Executed & Verified

---

## Hyperparameters & Strategy
- **Sampling Strategy**: Class Weighting (`class_weight="balanced"`, scale_pos_weight = 9.52)
- **Classifier**: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6)`
- **Training Positives**: 7,129
- **Training Negatives**: 67,871
- **Negative/Positive Ratio**: 9.52 : 1

---

## Validation Performance
- **Optimal Threshold**: `0.8400`
- **Validation Macro $F_0.5$ Score**: **`0.620184`**
- **Execution Runtime**: `67.53 seconds`
