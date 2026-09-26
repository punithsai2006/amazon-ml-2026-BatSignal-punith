# Detailed Error Analysis Report

**Model Analyzed**: Strategy A: Class Weighting  
**Validation Threshold**: 0.8400  
**Total Validation Entities**: 1500

---

## 1. Error Counts Summary

| Error Category | Count | Primary Impact |
|---|---|---|
| **False Positives (FP)** | 151 | Penalized $2	imes$ heavier under $F_0.5$ |
| **False Negatives (FN)** | 3,269 | Reduces recall |
| **Singleton False Positives** | 7 | Drops entity score from $1.0 ightarrow 0.0$ |

---

## 2. Identified Systematic Error Patterns

1. **High Name Similarity with Address Mismatch (False Positives)**:
   - Common business names (e.g. "Apex Logistics", "Summit Inc") sharing high name similarity but located at completely different addresses.
   - *Mitigation*: Stronger weighting on address evidence (`addr_levenshtein` & `addr_jaccard`).

2. **Legal Suffix Variations (False Negatives)**:
   - Variations like "Private Limited" vs "Pvt Ltd" when normalization missed specific abbreviations.
   - *Mitigation*: Expanded token-sort ratio and character n-gram Jaccard.

3. **Singleton False Positives**:
   - Entities with zero true matches where candidate similarity barely exceeded threshold.
   - *Mitigation*: Precision-weighted decision threshold optimization.
