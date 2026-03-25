import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

def tokenize(name: str) -> set:
    """Tokenize a string into a set of words. Returns an empty set if input is not a string."""
    if not isinstance(name, str):
        return set()
    return set(name.split())


def compute_features(row):
    """
    Compute a set of similarity and token-based features for a candidate pair of company names.
    Expects a row with 's_name_clean' and 'g_name_clean' columns.
    Returns a pandas Series with feature values.
    """
    # Extract cleaned names for S and G
    s = row["s_name_clean"]
    g = row["g_name_clean"]

    # Handle missing or non-string values gracefully
    if not isinstance(s, str):
        s = ""
    if not isinstance(g, str):
        g = ""

    # Tokenize both names into sets of words
    s_tokens = tokenize(s)
    g_tokens = tokenize(g)

    # Calculate token overlap (intersection and union)
    intersection = len(s_tokens & g_tokens)  # Number of shared tokens
    union = len(s_tokens | g_tokens)         # Total unique tokens across both names

    # Jaccard similarity: ratio of shared tokens to total unique tokens
    jaccard = intersection / union if union > 0 else 0

    # Length ratio: shorter name length divided by longer name length
    len_ratio = min(len(s), len(g)) / max(len(s), len(g)) if max(len(s), len(g)) > 0 else 0

    # Compute and return all features as a Series
    return pd.Series({
        # Fuzzy string similarity metrics (normalized to [0,1])
        # "lev_ratio": Levenshtein.normalized_similarity(s, g), # Normalized Levenshtein similarity
        "jaro_winkler": JaroWinkler.normalized_similarity(s, g), # Jaro-Winkler similarity
        "token_set_ratio": fuzz.token_set_ratio(s, g) / 100, # Token set ratio (compares shared words only)
        "token_sort_ratio": fuzz.token_sort_ratio(s, g) / 100, # Token sort ratio (sorts tokens alphabetically before comparing)
        "partial_ratio": fuzz.partial_ratio(s, g) / 100, # Partial ratio (looks for best matching substring)

        # Token-based features
        "jaccard": jaccard, # Jaccard similarity score

        # Length-based features
        "len_ratio": len_ratio, # Ratio of name lengths
        "len_diff": abs(len(s) - len(g)), # Absolute difference in name lengths
    })