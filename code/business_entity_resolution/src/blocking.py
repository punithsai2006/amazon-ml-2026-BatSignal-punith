"""
Multi-Stage Blocking Engine (BatSignal Team Baseline)
Generates candidate pairs using a union of blocking strategies
(token indexing, character n-gram TF-IDF similarity) and exports `candidate_pairs.tsv`.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from normalization import normalize_text
except ImportError:
    from src.normalization import normalize_text

logger = logging.getLogger(__name__)


class MultiStageBlocker:
    def __init__(self, top_k: int = 100, tfidf_min_df: int = 1):
        self.top_k = top_k
        self.tfidf_min_df = tfidf_min_df

    def _get_token_blocks(self, df: pd.DataFrame, id_col: str, text_col: str) -> Dict[str, Set[str]]:
        """Maps each unique token to the set of record IDs containing it."""
        token_to_ids: Dict[str, Set[str]] = {}
        for _, row in df.iterrows():
            rec_id = row[id_col]
            text = row.get(text_col, "")
            tokens = set(text.split())
            for token in tokens:
                if len(token) > 2:  # Skip trivial short tokens
                    if token not in token_to_ids:
                        token_to_ids[token] = set()
                    token_to_ids[token].add(rec_id)
        return token_to_ids

    def generate_candidates_for_source(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        target_source_label: str,
        s1_id_col: str = "s1_id",
        target_id_col: str = "s2_id"
    ) -> pd.DataFrame:
        """
        Generates candidate (s1_id, candidate_id) pairs between Source 1 and a Target Source (S2 or S3)
        using multi-strategy TF-IDF vector similarity & token indexing.
        """
        if s1_df.empty or target_df.empty:
            return pd.DataFrame(columns=["s1_id", "candidate_id", "source"])

        # Determine column names flexibly
        actual_s1_id_col = s1_id_col if s1_id_col in s1_df.columns else ("entity_id" if "entity_id" in s1_df.columns else s1_df.columns[0])
        actual_target_id_col = target_id_col if target_id_col in target_df.columns else ("entity_id" if "entity_id" in target_df.columns else target_df.columns[0])

        s1_name_col = "business_name_clean" if "business_name_clean" in s1_df.columns else "business_name"
        target_name_col = "business_name_clean" if "business_name_clean" in target_df.columns else "business_name"

        s1_texts = s1_df[s1_name_col].fillna("").tolist()
        target_texts = target_df[target_name_col].fillna("").tolist()

        # Vectorize using Character N-Grams for fuzzy match blocking
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=self.tfidf_min_df)
        all_texts = s1_texts + target_texts
        vectorizer.fit(all_texts)

        s1_vecs = vectorizer.transform(s1_texts)
        target_vecs = vectorizer.transform(target_texts)

        s1_ids = s1_df[actual_s1_id_col].values
        target_ids = target_df[actual_target_id_col].values

        candidates: List[Tuple[str, str, str]] = []

        # Compute pairwise similarity in batches for top-K extraction per S1 entity
        batch_size = 500
        for i in range(0, len(s1_ids), batch_size):
            batch_s1_vecs = s1_vecs[i : i + batch_size]
            batch_s1_ids = s1_ids[i : i + batch_size]
            sim_matrix = cosine_similarity(batch_s1_vecs, target_vecs)

            for idx, s1_id in enumerate(batch_s1_ids):
                row_sims = sim_matrix[idx]
                # Filter indices with similarity > 0.05
                valid_indices = np.where(row_sims > 0.05)[0]
                if len(valid_indices) == 0:
                    continue

                # Top-K indices
                if len(valid_indices) > self.top_k:
                    top_indices = valid_indices[np.argpartition(row_sims[valid_indices], -self.top_k)[-self.top_k:]]
                else:
                    top_indices = valid_indices

                for target_idx in top_indices:
                    candidates.append((s1_id, target_ids[target_idx], target_source_label))

        candidate_df = pd.DataFrame(candidates, columns=["s1_id", "candidate_id", "source"])
        logger.info(f"Generated {len(candidate_df)} candidate pairs for {target_source_label}")
        return candidate_df

    def generate_all_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame = None,
        output_path: str = "candidate_pairs.tsv"
    ) -> pd.DataFrame:
        """Generates all candidate pairs across Target Sources, saving to TSV."""
        cands_s2 = self.generate_candidates_for_source(s1_df, s2_df, "S2")
        cands_list = [cands_s2]

        if s3_df is not None and not s3_df.empty:
            cands_s3 = self.generate_candidates_for_source(s1_df, s3_df, "S3")
            cands_list.append(cands_s3)

        all_cands = pd.concat(cands_list, ignore_index=True)
        all_cands.drop_duplicates(subset=["s1_id", "candidate_id"], inplace=True)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            all_cands.to_csv(output_path, sep="\t", index=False)
            logger.info(f"Saved {len(all_cands)} candidate pairs to {output_path}")

        return all_cands


if __name__ == "__main__":
    s1 = pd.DataFrame([{"s1_id": "S1-1", "business_name": "Acme Solutions Pvt Ltd"}])
    s2 = pd.DataFrame([{"s2_id": "S2-101", "business_name": "Acme Solutions Private Limited"}])
    blocker = MultiStageBlocker(top_k=5)
    cands = blocker.generate_all_candidates(s1, s2, output_path="")
    print(cands)
