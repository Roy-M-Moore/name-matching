import pandas as pd
import numpy as np


def build_candidates(S_data, G_valid, indices, distances, include_labels=True):
    """Build candidate pairs based on similarity scores and adds company names.
    
    Parameters
    ----------
    S_data : pd.DataFrame
        Source data with columns: train_index (or test_index), name_clean, 
        and optionally company_id for labeled data
    G_valid : pd.DataFrame
        Target data with columns: company_id, name_clean
    indices : np.ndarray
        Array of shape (n_samples, n_neighbors) with G indices for nearest neighbors
    distances : np.ndarray
        Array of shape (n_samples, n_neighbors) with cosine distances
    include_labels : bool, default=True
        Whether to include true labels and match column (for training data)
    
    Returns
    -------
    pd.DataFrame
        Candidate pairs with columns:
        - train_index/test_index: index from S_data
        - candidate_company_id: company_id from G
        - tfidf_similarity: 1 - cosine distance
        - s_name_clean: clean name from S_data
        - g_name_clean: clean name from G
        - true_company_id: (optional) true company_id from S_data
        - match: (optional) binary indicator of true match
    """
    # Determine index column name
    index_col = "train_index" if "train_index" in S_data.columns else "test_index"
    
    # Vectorized candidate pair generation
    n_samples, n_neighbors = indices.shape
    
    # Create flattened arrays for all candidate pairs
    s_indices = np.repeat(S_data[index_col].values, n_neighbors)
    g_row_indices = indices.flatten()
    similarities = (1 - distances).flatten()
    g_ids = G_valid.iloc[g_row_indices]["company_id"].values
    
    # Build candidates DataFrame in one go
    candidates = pd.DataFrame({
        index_col: s_indices,
        "candidate_company_id": g_ids,
        "tfidf_similarity": similarities
    })
    
    # Prepare merge columns based on whether labels are included
    s_merge_cols = [index_col, "name_clean"]
    if include_labels:
        s_merge_cols.insert(1, "company_id")
    
    # Merge S_data and G_valid information
    candidates = (
        candidates
        .merge(
            S_data[s_merge_cols],
            on=index_col,
            how="left"
        )
        .merge(
            G_valid[["company_id", "name_clean"]],
            left_on="candidate_company_id",
            right_on="company_id",
            how="left",
            suffixes=("_s", "_g")
        )
        .rename(columns={
            "name_clean_s": "s_name_clean",
            "name_clean_g": "g_name_clean"
        })
    )
    
    # Handle label-specific columns
    if include_labels:
        candidates["match"] = (
            candidates["candidate_company_id"] == candidates["company_id_s"]
        ).astype(int)
        candidates = candidates.rename(columns={"company_id_s": "true_company_id"})
        candidates = candidates.drop(columns=["company_id_g"])
    else:
        # For test data, drop the extra company_id column
        candidates = candidates.drop(columns=["company_id"])
    
    return candidates
