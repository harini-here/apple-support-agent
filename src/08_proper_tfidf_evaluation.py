import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "apple_customers.csv"
)

TEST_FILE = (
    BASE_DIR
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "proper_tfidf_results.csv"
)

# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("Loading training data...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Training examples: {len(train):,}")
print(f"Golden test examples: {len(test):,}")


# --------------------------------------------------
# Clean training data
# --------------------------------------------------

train["clean_text"] = train["text"].fillna("").astype(str)

test["clean_text"] = test["clean_text"].fillna("").astype(str)

# IMPORTANT:
# Remove Golden Set examples from training if any
# tweet IDs overlap.

if "tweet_id" in train.columns and "tweet_id" in test.columns:

    test_ids = set(test["tweet_id"])

    train = train[
        ~train["tweet_id"].isin(test_ids)
    ].copy()

print(
    f"Training examples after removing test overlap: "
    f"{len(train):,}"
)


# --------------------------------------------------
# Labels
# --------------------------------------------------

X_train = train["clean_text"]
y_train = train["golden_intent"] if "golden_intent" in train.columns else None

# The raw Apple dataset does NOT contain intent labels.
# Therefore we need pseudo-labels for training.
#
# For this assignment we use the candidate labeling
# pipeline to create training labels separately from
# the Golden Set.
#
# For now, use the candidate-intent file as our labeled
# training source.

CANDIDATE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_candidates.csv"
)

candidate = pd.read_csv(CANDIDATE_FILE)

candidate["clean_text"] = (
    candidate["clean_text"]
    .fillna("")
    .astype(str)
)

# --------------------------------------------------
# Train on candidate-labeled examples
# --------------------------------------------------

X_train = candidate["clean_text"]
y_train = candidate["candidate_intent"]

print(
    f"Training examples with labels: {len(X_train):,}"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
    )
])


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# Test on Golden Set
# --------------------------------------------------

X_test = test["clean_text"]
y_test = test["golden_intent"]

predictions = model.predict(X_test)

test["predicted_intent"] = predictions


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n================================")
print("PROPER TF-IDF EVALUATION")
print("================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# --------------------------------------------------
# Detailed report
# --------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# --------------------------------------------------
# Save
# --------------------------------------------------

test.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)