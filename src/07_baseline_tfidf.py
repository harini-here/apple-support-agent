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

INPUT_FILE = BASE_DIR / "evaluation" / "golden_set.csv"
OUTPUT_FILE = BASE_DIR / "evaluation" / "baseline_tfidf_results.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Golden Set size: {len(df)}")

X = df["clean_text"].fillna("")
y = df["golden_intent"]

# --------------------------------------------------
# TF-IDF + Logistic Regression
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

print("\nTraining TF-IDF + Logistic Regression...")

model.fit(X, y)

# --------------------------------------------------
# Predict
# --------------------------------------------------

predictions = model.predict(X)

df["predicted_intent"] = predictions

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

accuracy = accuracy_score(y, predictions)

macro_f1 = f1_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y,
    predictions,
    average="weighted",
    zero_division=0
)

print("\n================================")
print("BASELINE #2 — TF-IDF + LOGISTIC")
print("================================")

print(f"Accuracy   : {accuracy:.4f}")
print(f"Macro F1   : {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y,
        predictions,
        zero_division=0
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

labels = sorted(
    set(y) | set(predictions)
)

cm = confusion_matrix(
    y,
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
# Save results
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_FILE)