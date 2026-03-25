import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

from src import preprocessing
from src import feature_engineering
from src import thresholding
from src import candidate_generation

import time
start_time = time.time()

### PRE-PROCESSING
# Read data
G = pd.read_csv('input/G.csv', sep = '|')
STrain = pd.read_csv('input/STrain.csv', sep = '|')

# Clean company names
G["name_clean"] = G["name"].apply(preprocessing.preprocess_company_name)
STrain["name_clean"] = STrain["name"].apply(preprocessing.preprocess_company_name)
STrain = STrain.iloc[0:10000] ###########/////////////// TESTING ONLY

# Identify problematic entries
G_problems = ((G['name_clean'].str.len() == 0) | (G['name_clean'].duplicated(keep=False)))

# Create a clean version for matching
G_valid = G[~G_problems].copy()
G_invalid = G[G_problems].copy() 

### BLOCKING
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3,5),
    min_df=2
)

G_tfidf = tfidf.fit_transform(G_valid["name_clean"])
S_tfidf = tfidf.transform(STrain["name_clean"])

# Save objects for re-use during inference
# joblib.dump(tfidf, 'models/tfidf_vectorizer.joblib') ###################################
# joblib.dump(G_tfidf, 'models/G_tfidf.joblib') ###################################

# Fit nearest neighbors model for blocking
enn = NearestNeighbors(
    n_neighbors=50,
    metric='cosine',
    algorithm='brute',
    n_jobs=-1
)

enn.fit(G_tfidf)

distances, indices = enn.kneighbors(S_tfidf)

# Generate candidate pairs from nearest neighbors blocking results
candidates = candidate_generation.build_candidates(
    S_data=STrain,
    G_valid=G_valid,
    indices=indices,
    distances=distances,
    include_labels=True
)

# Calculate & print blocking recall
recall = candidates[candidates["match"] == 1].shape[0] / \
         STrain[STrain["company_id"] != -1].shape[0]

print(f"Blocking recall: {recall}")

sys.exit(1) ######### STOP AFTER BLOCKING FOR NOW ############

### FEATURE ENGINEERING
feature_df = candidates.apply(
    feature_engineering.compute_features,
    axis=1
)

# Create training df
train_df = pd.concat(
    [candidates, feature_df],
    axis=1
)

### TRAIN MODEL
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

X = train_df[feature_cols]
y = train_df["match"]

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'scale_pos_weight': [0.2, 0.5, 1, 2, 5],
    'max_depth': [8, 10, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [200, 300, 500],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2],
}

search = RandomizedSearchCV(
    XGBClassifier(random_state=42),
    param_dist, n_iter=50, cv=3, scoring='roc_auc', n_jobs=-1
)

search.fit(X, y)

# Train final model with best hyperparameters
xgb_model = XGBClassifier(
    random_state=42,
    **search.best_params_
)

xgb_model.fit(X, y)

# Save the trained model for inference
joblib.dump(xgb_model, 'models/trained_model.joblib')

# Predict probabilities on the training set (for threshold tuning)
xgb_probs = xgb_model.predict_proba(X)[:,1]

# Print training ROC_AUC
print(f"Training set ROC_AUC: {roc_auc_score(y, xgb_probs)}")

### THRESHOLD TUNING
train_df["match_prob"] = xgb_probs

best_candidates = (
    train_df
    .sort_values("match_prob", ascending=False)
    .groupby("train_index")
    .first()
    .reset_index()
)

# Sweep thresholds to evaluate cost, FP, FN, TP at each value
results = []
thresholds = np.linspace(0,1,200)

for t in thresholds:
    # Compute cost and error counts for each threshold
    cost, fp, fn, tp = thresholding.compute_cost(best_candidates, t)
    results.append([t, cost, fp, fn, tp])

cost_df = pd.DataFrame(
    results,
    columns=["threshold","cost","fp","fn","tp"]
)

# Get optimal threshold that minimizes cost
best_row = cost_df.loc[cost_df["cost"].idxmin()]
t_star = best_row["threshold"]

# Save the optimal threshold for inference
joblib.dump(t_star, 'models/threshold.joblib')

# Print the optimal threshold and training cost at that threshold
print(f"Optimal threshold (t*): {t_star}")
print(f"Cost at optimal threshold: {best_row['cost']}")

end_time = time.time()
print(f"\nTotal runtime: {(end_time - start_time)/60:.2f} minutes", flush=True)