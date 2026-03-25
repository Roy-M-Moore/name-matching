import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors


from src import preprocessing
from src import feature_engineering
from src import candidate_generation

import time
start_time = time.time()

# Accept STest file path from command line
if len(sys.argv) < 2:
    print("Usage: python run.py <path_to_stest_file>")
    sys.exit(1)

stest_path = sys.argv[1]

# Read data
G = pd.read_csv('input/G.csv', sep = '|')
STest = pd.read_csv(stest_path, sep='|')

# Validate STest format
if not {'test_index', 'name'}.issubset(STest.columns):
    print("Error: Input file must contain 'test_index' and 'name' columns")
    sys.exit(1)

# Clean company names
G["name_clean"] = G["name"].apply(preprocessing.preprocess_company_name)
STest["name_clean"] = STest["name"].apply(preprocessing.preprocess_company_name)

# Identify problematic entries
G_problems = ((G['name_clean'].str.len() == 0) | (G['name_clean'].duplicated(keep=False)))

# Create a clean version for matching
G_valid = G[~G_problems].copy()
G_invalid = G[G_problems].copy() 

### BLOCKING
# Load pre-computed TF-IDF vectorizer and G_tfidf matrix
tfidf = joblib.load('models/tfidf_vectorizer.joblib')
G_tfidf = joblib.load('models/G_tfidf.joblib')


S_tfidf = tfidf.transform(STest["name_clean"])

# Fit nearest neighbors model for blocking
enn = NearestNeighbors(
    n_neighbors=25,
    metric='cosine',
    algorithm='brute',
    n_jobs=-1
)

enn.fit(G_tfidf)

distances, indices = enn.kneighbors(S_tfidf)

# Generate candidate pairs from nearest neighbors blocking results
candidates = candidate_generation.build_candidates(
    S_data=STest,
    G_valid=G_valid,
    indices=indices,
    distances=distances,
    include_labels=False
)

### FEATURE ENGINEERING
feature_df = candidates.apply(
    feature_engineering.compute_features,
    axis=1
)

# Create test df
test_df = pd.concat(
    [candidates, feature_df],
    axis=1
)

### RUN MODEL
feature_cols = [
    "tfidf_similarity",
    "jaro_winkler",
    "token_set_ratio",
    "token_sort_ratio",
    "partial_ratio",
    "jaccard",
    "len_ratio",
    "len_diff"
]

X = test_df[feature_cols]

# Load trained model
xgb_model = joblib.load('models/trained_model.pkl') ###################################

# Get predicted probabilities
xgb_probs = xgb_model.predict_proba(X)[:,1]


### PREDICTIONS
test_df["match_prob"] = xgb_probs

best_candidates = (
    test_df
    .sort_values("match_prob", ascending=False)
    .groupby("test_index")
    .first()
    .reset_index()
)

# Load the tuned threshold
t_star = joblib.load('models/threshold.joblib')

# Apply threshold to make final predictions
best_candidates["company_id"] = np.where(
    best_candidates["match_prob"] >= t_star,
    best_candidates["candidate_company_id"],
    -1  # No match
)

# Create submission
submission = best_candidates[["test_index", "company_id"]]
submission.to_csv("output/submission.csv", sep='|', index=False)

end_time = time.time()
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes", flush=True)