# Candidate Blocking Class Imbalance Report

**Team**: BatSignal  
**Date**: 2026-09-26 17:20:16  
**Validation Split**: `splits/val_entity_ids.txt` (Strict Entity-Level Isolation)

---

## Candidate Blocking Class Imbalance Summary

| Dataset Portion | Total Candidate Pairs | Positive Matches | Negative Candidates | Positive Fraction | Negative Fraction | Imbalance Ratio (N:P) |
|---|---|---|---|---|---|---|
| **Training Set** | 75,000 | 7,129 | 67,871 | 0.0951 | 0.9049 | **9.52 : 1** |
| **Validation Set** | 22,500 | 2,008 | 20,492 | 0.0892 | 0.9108 | **10.21 : 1** |

### Key Findings:
1. Blocking generates **9.52 negative candidate pairs for every positive match**.
2. Training and validation portions exhibit nearly identical class imbalance ratios, confirming consistent candidate generation across entity splits.
