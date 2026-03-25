import pandas as pd
import numpy as np

def compute_cost(df, threshold):
    # Determine which rows are predicted as matches based on the threshold
    pred_match = df["match_prob"] >= threshold
    
    # Assign predicted company_id: if predicted as match, use candidate_company_id; else -1
    pred_id = np.where(pred_match, df["candidate_company_id"], -1)
    
    # True company_id for each row
    true_id = df["true_company_id"]
    
    # Count false positives
    fp = np.sum(
        (pred_id != -1) &
        (pred_id != true_id)
    )
    
    # Count false negatives
    fn = np.sum(
        (pred_id == -1) &
        (true_id != -1)
    )
    
    # Count true positives
    tp = np.sum(
        (pred_id != -1) &
        (pred_id == true_id)
    )

    total_cost = 5*fp + 1*fn
    
    return total_cost, fp, fn, tp