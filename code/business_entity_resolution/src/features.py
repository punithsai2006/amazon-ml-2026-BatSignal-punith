"""
Pairwise Feature Engineering Engine (BatSignal Team Baseline)
Extracts string metrics, token similarities, structural features,
and country match indicators for candidate (S1, Target) pairs.
"""

import math
import logging
import pandas as pd
import numpy as np
from typing import Dict, List
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def levenshtein_ratio(str1: str, str2: str) -> float:
    """Computes normalized sequence similarity ratio [0, 1]."""
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1, str2).ratio()


def token_jaccard(str1: str, str2: str) -> float:
    """Computes Jaccard similarity over word tokens."""
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / float(len(union))


def token_sort_ratio(str1: str, str2: str) -> float:
    """Computes Levenshtein ratio on sorted tokens."""
    sorted1 = " ".join(sorted(str1.split()))
    sorted2 = " ".join(sorted(str2.split()))
    return levenshtein_ratio(sorted1, sorted2)


def char_ngram_jaccard(str1: str, str2: str, n: int = 3) -> float:
    """Computes Jaccard similarity over character n-grams."""
    if len(str1) < n or len(str2) < n:
        return levenshtein_ratio(str1, str2)
    ngrams1 = set(str1[i : i + n] for i in range(len(str1) - n + 1))
    ngrams2 = set(str2[i : i + n] for i in range(len(str2) - n + 1))
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / float(len(union)) if union else 0.0


def extract_pair_features(row: pd.Series) -> Dict[str, float]:
    """
    Extracts complete pairwise feature vector for a single candidate record.
    Degrades safely on empty/missing values (never producing NaNs).
    """
    name1 = str(row.get("business_name_s1", "")) if not pd.isna(row.get("business_name_s1")) else ""
    name2 = str(row.get("business_name_cand", "")) if not pd.isna(row.get("business_name_cand")) else ""

    addr1 = str(row.get("business_address_s1", "")) if not pd.isna(row.get("business_address_s1")) else ""
    addr2 = str(row.get("business_address_cand", "")) if not pd.isna(row.get("business_address_cand")) else ""

    country1 = str(row.get("country_s1", "")).lower() if not pd.isna(row.get("country_s1")) else ""
    country2 = str(row.get("country_cand", "")).lower() if not pd.isna(row.get("country_cand")) else ""

    # Name similarity metrics
    name_lev = levenshtein_ratio(name1, name2)
    name_jaccard = token_jaccard(name1, name2)
    name_sort = token_sort_ratio(name1, name2)
    name_ngram = char_ngram_jaccard(name1, name2, n=3)
    name_len_diff = abs(len(name1) - len(name2))
    name_len_ratio = min(len(name1), len(name2)) / max(len(name1), len(name2), 1)

    # Address similarity metrics
    addr_lev = levenshtein_ratio(addr1, addr2)
    addr_jaccard = token_jaccard(addr1, addr2)
    addr_ngram = char_ngram_jaccard(addr1, addr2, n=3)
    addr_len_ratio = min(len(addr1), len(addr2)) / max(len(addr1), len(addr2), 1)

    # Country matching feature (Never hard-filter, open-set handling)
    country_match = 1.0 if (country1 and country2 and country1 == country2) else 0.0
    country_missing = 1.0 if not country1 or not country2 else 0.0

    # Combined composite score
    composite_sim = (0.6 * name_lev) + (0.4 * addr_lev)

    features = {
        "name_levenshtein": name_lev,
        "name_jaccard": name_jaccard,
        "name_token_sort": name_sort,
        "name_ngram_jaccard": name_ngram,
        "name_length_diff": float(name_len_diff),
        "name_length_ratio": name_len_ratio,
        "addr_levenshtein": addr_lev,
        "addr_jaccard": addr_jaccard,
        "addr_ngram_jaccard": addr_ngram,
        "addr_length_ratio": addr_len_ratio,
        "country_match": country_match,
        "country_missing": country_missing,
        "composite_sim": composite_sim,
    }

    return features


def build_feature_matrix(
    candidate_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Builds a full feature matrix DataFrame for all candidate pairs."""
    if candidate_pairs.empty:
        return pd.DataFrame()

    logger.info("Merging entity details onto candidate pairs...")

    s1_id_col = "s1_id" if "s1_id" in candidate_pairs.columns else "source1_entity_id"
    cand_id_col = "candidate_id" if "candidate_id" in candidate_pairs.columns else "candidate_entity_ids"

    s1_entity_col = "entity_id" if "entity_id" in s1_df.columns else s1_df.columns[0]
    
    # Map S1 attributes
    merged = candidate_pairs.merge(
        s1_df, left_on=s1_id_col, right_on=s1_entity_col, how="left"
    ).rename(columns={
        "business_name": "business_name_s1",
        "business_address": "business_address_s1",
        "country": "country_s1"
    })

    # Prepare candidate pools for S2 and S3
    s2_id_col = "entity_id" if "entity_id" in s2_df.columns else s2_df.columns[0]
    s2_renamed = s2_df.rename(columns={
        s2_id_col: cand_id_col,
        "business_name": "business_name_cand",
        "business_address": "business_address_cand",
        "country": "country_cand"
    })

    cand_pools = [s2_renamed]

    if s3_df is not None and not s3_df.empty:
        s3_id_col = "entity_id" if "entity_id" in s3_df.columns else s3_df.columns[0]
        s3_renamed = s3_df.rename(columns={
            s3_id_col: cand_id_col,
            "business_name": "business_name_cand",
            "business_address": "business_address_cand",
            "country": "country_cand"
        })
        cand_pools.append(s3_renamed)

    cand_pool = pd.concat(cand_pools, ignore_index=True).drop_duplicates(subset=[cand_id_col])

    merged = merged.merge(cand_pool, on=cand_id_col, how="left")

    logger.info("Extracting pairwise features for candidates...")
    feature_dicts = merged.apply(extract_pair_features, axis=1)
    feature_df = pd.DataFrame(list(feature_dicts))

    source_col = "source" if "source" in merged.columns else ("candidate_source" if "candidate_source" in merged.columns else None)
    id_cols = [s1_id_col, cand_id_col]
    if source_col and source_col in merged.columns:
        id_cols.append(source_col)

    result_df = pd.concat([merged[id_cols], feature_df], axis=1)
    return result_df


if __name__ == "__main__":
    row_sample = pd.Series({
        "business_name_s1": "Acme Industrial Pvt Ltd",
        "business_name_cand": "Acme Industrial Private Limited",
        "business_address_s1": "100 MG Rd Bangalore",
        "business_address_cand": "100 MG Road Bengaluru",
        "country_s1": "India",
        "country_cand": "India"
    })
    print("Sample Features:", extract_pair_features(row_sample))
