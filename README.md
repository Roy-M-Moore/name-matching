# Name Matching

A machine learning pipeline for entity resolution of company names. This system matches potentially misspelled or variant company names from a source dataset to a canonical reference dataset using TF-IDF blocking, feature engineering, and XGBoost classification.

## Problem Framing
This is a supervised binary classification problem for entity resolution based exclusively on name matching.

- **Asymmetric misclassification cost**: The problem has a 5:1 cost ratio, where false positives (incorrect matches) are 5× more expensive than false negatives (missed matches). This reflects the business reality that incorrectly linking two distinct legal entities is more harmful than failing to identify a match.
- **Precision over recall** : Given the asymmetric cost structure, the system prioritizes high precision over recall: it's better to leave a name unmatched than to link the wrong entities.
- **Decision threshold**: Final linking decisions use an optimized probability threshold that minimizes the expected matching cost (5×FP + 1×FN) on the training set.

## Project Overview

The package implements a four-stage pipeline for fuzzy company name matching:

1. **Preprocessing**: Normalizes company names (Unicode normalization, lowercasing, legal suffix removal)
2. **Blocking**: Uses TF-IDF vectorization + k-NN to efficiently generate candidate pairs
3. **Feature Engineering**: Computes multiple similarity metrics between candidate pairs
4. **Classification**: XGBoost model predicts match probabilities, with optimized threshold tuning

## Project Structure

```
name-matching/
├── input/                    # Input data files
│   ├── G.csv                # Reference company dataset (canonical names)
│   ├── STrain.csv           # Training dataset with labeled matches
│   └── STest.csv            # Test dataset for predictions
├── models/                   # Saved model artifacts
│   ├── tfidf_vectorizer.joblib
│   ├── G_tfidf.joblib
│   ├── trained_model.joblib
│   └── threshold.joblib
├── output/
│   └── submission.csv       # Prediction output
├── src/                     # Core modules
│   ├── preprocessing.py     # Name normalization functions
│   ├── candidate_generation.py  # Candidate pair generation
│   ├── feature_engineering.py   # Similarity feature computation
│   └── thresholding.py      # Cost-based threshold optimization
├── train.py                 # Model training pipeline
└── run.py                   # Inference pipeline
```

## Installation

```bash
pip install -e .
```

**Dependencies:**
- pandas, numpy
- scikit-learn
- xgboost
- rapidfuzz
- unidecode
- regex
- joblib

## Usage

### Training

Train the model on labeled data:

```bash
python train.py
```

This will:
- Load and preprocess `G.csv` and `STrain.csv`
- Generate candidate pairs using TF-IDF blocking
- Compute similarity features for each candidate pair
- Train an XGBoost classifier
- Optimize the classification threshold to minimize a custom cost function (5×FP + 1×FN)
- Save trained artifacts to models

**Output metrics:**
- Blocking recall (percentage of true matches in candidate set)
- Training ROC-AUC
- Optimal threshold and cost

### Inference

Make predictions on new data:

```bash
python run.py <path_to_test_file>
```

Example:
```bash
python run.py input/STest.csv
```

**Input format:** Pipe-separated ('|') CSV with columns `test_index` and `name`

**Output:** submission.csv with columns:
- `test_index`: Test record identifier
- `company_id`: Matched company ID from G.csv, or `-1` for no match

## Pipeline Details

### 1. Preprocessing (preprocessing.py)

Normalizes company names through:
- **Unicode normalization**: Converts accented characters (e.g., `müller` → `muller`)
- **Lowercasing**: Standardizes case
- **Initial handling**: `A.B.C. Inc` → `ABC Inc`
- **Legal suffix removal**: Removes common entity types (`LLC`, `Inc`, `Ltd`, `GmbH`, etc.)
- **Punctuation handling**: Converts to whitespace

### 2. Blocking (train.py, run.py)

Reduces the search space by:
- TF-IDF vectorization with character n-grams (3-5 grams)
- k-Nearest Neighbors (k=25 for inference) using cosine similarity
- Only generates candidates from the top-k most similar reference names

### 3. Feature Engineering (feature_engineering.py)

Computes 8 similarity features for each candidate pair:

| Feature | Description |
|---------|-------------|
| `tfidf_similarity` | Cosine similarity from TF-IDF (1 - distance) |
| `jaro_winkler` | Jaro-Winkler string similarity |
| `token_set_ratio` | Token-based similarity (order-independent) |
| `token_sort_ratio` | Token-based similarity (sorted) |
| `partial_ratio` | Best matching substring similarity |
| `jaccard` | Jaccard similarity of word tokens |
| `len_ratio` | Ratio of shorter/longer name length |
| `len_diff` | Absolute difference in name lengths |

### 4. Classification & Thresholding

- **Model**: XGBoost binary classifier
- **Threshold optimization**: Sweeps thresholds from 0 to 1 to minimize custom cost function
- **Cost function**: `5 × (false positives) + 1 × (false negatives)`
- **Prediction logic**: For each test record, select the highest-probability candidate; assign match if probability ≥ threshold


## Performance Considerations

- **Blocking recall**: Typically ~85%
- **XGBoost ROC AUC**: Typically ~99% on the training set
- **Inference Runtime**: ~1 minute per 10K records [tested on HP Elitebook 845 G10: Ryzen 5 PRO 7545U (3.20GHz) / 32 GB RAM]
- **Threshold value**: Threshold optimization typically results in a probability threshold of ~.40
- **Cost value**: Optimized cost value on the final training data run was ~41.5k

## Potential Improvements
- Improve blocking recall: explore multi-stage blocking, improved preprocessing, alternative distance metrics
- Implement train/validation split for more robust threshold tuning
- Integrate custom cost function into hyperparameter optimization (currently applied only at threshold selection)