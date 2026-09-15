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
    / "apple_training.csv"
)

TEST_FILE = (
    BASE_DIR
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "tfidf_final_results.csv"
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading datasets...")

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"Training examples : {len(train):,}")
print(f"Golden test       : {len(test):,}")

# --------------------------------------------------
# Prepare data
# --------------------------------------------------

X_train = train["clean_text"].fillna("")
y_train = train["intent"]

X_test = test["clean_text"].fillna("")
y_test = test["golden_intent"]

# --------------------------------------------------
# Build model
# --------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            max_features=100000
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

print("\nTraining TF-IDF + Logistic Regression...")

model.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X_test)

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
print("BASELINE #2 — PROPER EVALUATION")
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
# Confusion matrix
# --------------------------------------------------

labels = sorted(
    set(y_test) | set(predictions)
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)

# --------------------------------------------------
# Save predictions
# --------------------------------------------------

results = test.copy()

results["predicted_intent"] = predictions

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)