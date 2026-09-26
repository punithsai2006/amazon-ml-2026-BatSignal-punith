# BatSignal Experiment Report: Class-Weighting vs. Hard-Negative Mining

**Team**: BatSignal  
**Author**: Task Assignment (Class-Weighting vs Hard-Negative Mining)  
**Date**: 26 Sept 2026  
**Status**: Metric Verified & Experiment Pipeline Ready  

---

## 1. Metric Verification Status (Nandini & Team Audit)

The official $F_{0.5}$ metric computation was verified in `code/business_entity_resolution/src/verify_metric.py` against the prompt's worked example:

- **Worked Example Test**:
  - Prediction: `S1-00001` $\rightarrow$ `[S2-00047, S2-00193, S3-00812]`
  - Ground Truth: `S1-00001` $\rightarrow$ `[S2-00047, S3-00812]`
  - Formula: $$F_{0.5} = \frac{1.25 \times P \times R}{0.25 \times P + R} = \frac{1.25 \times (2/3) \times 1.0}{0.25 \times (2/3) + 1.0} = 0.7142857$$
  - **Result**: **PASS** ($0.714286$, exact match with $0.714$).

- **Singleton Rules Test**:
  - True Singleton + empty prediction = $1.0$ (**PASS**).
  - True Singleton + false match = $0.0$ (**PASS**).

---

## 2. Dedicated Experiment Pipeline (`exp_class_weight_vs_hard_negatives.py`)

To adhere strictly to team protocol and prevent overwriting Vamsika's baseline files, the experimental harness has been isolated into:
`code/business_entity_resolution/src/exp_class_weight_vs_hard_negatives.py`

### Key Functions Implemented:
1. `load_validation_entity_ids()`: Enforces validation scoring strictly against the team's shared split file `splits/val_entity_ids.txt`.
2. `measure_class_imbalance()`: Measures and logs the exact positive-to-negative candidate pair ratio produced by `blocking.py`.
3. `run_experiment_comparison()`: Executes a controlled comparison between:
   - **Strategy (a)**: Class-weighting (`scale_pos_weight` / balanced sample weights).
   - **Strategy (b)**: Explicit hard-negative mining (selecting non-matches sorted by top similarity scores).

---

## 3. Immediate Action Required: Dataset Download & Placement

To run the experiment and log final validation numbers:

1. Download the **Student Resource** dataset from Unstop:
   `https://unstop.com/competitions/1743604/round/1593683/play/code`
2. Extract the `student_resource/` folder.
3. Copy the `dataset/` directory contents into the repository:
   ```bash
   cp -r student_resource/dataset/* amazon-ml-2026-BatSignal/dataset/
   ```
4. Confirm the 7 TSV files exist at:
   - `dataset/train/train_source1.tsv`
   - `dataset/train/train_source2.tsv`
   - `dataset/train/train_source3.tsv`
   - `dataset/train/train_ground_truth.tsv`
   - `dataset/test/test_source1.tsv`
   - `dataset/test/test_source2.tsv`
   - `dataset/test/test_source3.tsv`
5. Run the experiment comparison:
   ```bash
   python code/business_entity_resolution/src/exp_class_weight_vs_hard_negatives.py
   ```
