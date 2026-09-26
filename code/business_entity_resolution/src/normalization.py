"""
Text Normalization & Canonicalization Module (BatSignal Team Baseline)
Handles Unicode normalization, lowercasing, punctuation stripping,
legal suffix canonicalization, and address abbreviation expansion.
"""

import re
import unicodedata
import pandas as pd
from typing import Optional

LEGAL_SUFFIX_MAP = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\benterprises\b": "enterprise",
    r"\bsolutions\b": "solution",
    r"\bservices\b": "service"
}

ADDRESS_ABBREV_MAP = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bapt\b": "apartment",
    r"\bbldg\b": "building",
    r"\bfl\b": "floor",
    r"\bno\b": "number",
    r"\bste\b": "suite",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bpkwy\b": "parkway"
}


def clean_unicode(text: str) -> str:
    """Normalizes Unicode characters and removes accent marks."""
    if not isinstance(text, str) or not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_text(text: Optional[str], is_address: bool = False) -> str:
    """
    Performs full text normalization pipeline:
    1. Unicode decomposition
    2. Lowercasing
    3. Punctuation removal
    4. Suffix / abbreviation canonicalization
    5. Whitespace collapsing
    """
    if text is None or pd.isna(text):
        return ""
    
    text = clean_unicode(str(text))
    text = text.lower()
    
    # Replace non-alphanumeric characters with space (except keeping spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Canonicalize legal suffixes & address abbreviations
    mapping = {**LEGAL_SUFFIX_MAP, **ADDRESS_ABBREV_MAP} if is_address else LEGAL_SUFFIX_MAP
    for pattern, replacement in mapping.items():
        text = re.sub(pattern, replacement, text)
        
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_dataframe(df: pd.DataFrame, name_col: str = "business_name", address_col: str = "business_address") -> pd.DataFrame:
    """Applies text normalization across specified columns of a DataFrame."""
    df_clean = df.copy()
    if name_col in df_clean.columns:
        df_clean[f"{name_col}_clean"] = df_clean[name_col].apply(lambda x: normalize_text(x, is_address=False))
    if address_col in df_clean.columns:
        df_clean[f"{address_col}_clean"] = df_clean[address_col].apply(lambda x: normalize_text(x, is_address=True))
    return df_clean


if __name__ == "__main__":
    sample_name = "Acme Ind. Solutions Pvt. Ltd."
    sample_addr = "100 MG Rd., Ste 4B"
    print("Normalized Name:", normalize_text(sample_name))
    print("Normalized Address:", normalize_text(sample_addr, is_address=True))
