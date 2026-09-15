import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading datasets...")

train_df = pd.read_csv(
    "data/processed/apple_training.csv"
)

golden_df = pd.read_csv(
    "evaluation/golden_set.csv"
)

train_df["clean_text"] = train_df["clean_text"].fillna("")
golden_df["clean_text"] = golden_df["clean_text"].fillna("")

print(f"Training examples : {len(train_df):,}")
print(f"Golden examples   : {len(golden_df):,}")


# ============================================================
# 2. TRAIN TF-IDF + LOGISTIC REGRESSION
# ============================================================

print("\nTraining TF-IDF classifier...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_df["clean_text"]
)

y_train = train_df["intent"]

classifier = LogisticRegression(
    class_weight="balanced",
    max_iter=2000
)

classifier.fit(
    X_train,
    y_train
)

print("Classifier ready.")


# ============================================================
# 3. PREDICT GOLDEN SET
# ============================================================

X_test = vectorizer.transform(
    golden_df["clean_text"]
)

probabilities = classifier.predict_proba(
    X_test
)

predicted_indices = np.argmax(
    probabilities,
    axis=1
)

predicted_intents = classifier.classes_[
    predicted_indices
]

confidence = np.max(
    probabilities,
    axis=1
)

golden_df["predicted_intent"] = predicted_intents
golden_df["confidence"] = confidence


# ============================================================
# 4. INTENT CLASSIFICATION EVALUATION
# ============================================================

y_true = golden_df["golden_intent"]
y_pred = golden_df["predicted_intent"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n================================")
print("INTENT CLASSIFICATION")
print("================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 5. ESCALATION / RISK RULES
# ============================================================

ESCALATION_PATTERNS = {

    "refund_or_charge": [
        "refund",
        "charged twice",
        "charged me",
        "wrong charge",
        "unauthorized charge",
        "money back",
        "payment dispute"
    ],

    "account_security": [
        "hacked",
        "account stolen",
        "someone accessed",
        "security breach",
        "unauthorized access",
        "can't access my account"
    ],

    "legal_or_serious": [
        "lawsuit",
        "legal action",
        "court",
        "police",
        "fraud",
        "scam"
    ],

    "hardware_service": [
        "genius bar",
        "repair",
        "replacement",
        "broken",
        "physical damage"
    ]
}


def detect_risk(text):

    text = str(text).lower()

    for reason, patterns in ESCALATION_PATTERNS.items():

        for pattern in patterns:

            if pattern in text:
                return True, reason

    return False, None


# ============================================================
# 6. AUTO-HANDLE / ESCALATION DECISION
# ============================================================

THRESHOLD = 0.55

predicted_actions = []
escalation_reasons = []

for text, prob in zip(
    golden_df["clean_text"],
    confidence
):

    risky, risk_reason = detect_risk(text)

    if risky:

        action = "ESCALATE"
        reason = risk_reason

    elif prob < THRESHOLD:

        action = "ESCALATE"
        reason = "low_confidence"

    else:

        action = "AUTO_HANDLE"
        reason = "high_confidence"

    predicted_actions.append(action)
    escalation_reasons.append(reason)


golden_df["predicted_action"] = predicted_actions

golden_df["predicted_escalation_reason"] = (
    escalation_reasons
)


# ============================================================
# 7. AUTO-HANDLE / ESCALATION EVALUATION
# ============================================================

print("\n================================")
print("AUTO-HANDLE / ESCALATION")
print("================================")


if "expected_action" in golden_df.columns:

    # Remove missing expected_action values
    action_df = golden_df[
        golden_df["expected_action"].notna()
    ].copy()

    # Remove empty strings
    action_df = action_df[
        action_df["expected_action"]
        .astype(str)
        .str.strip()
        != ""
    ]

    print(
        f"Rows with expected action: {len(action_df)}"
    )

    if len(action_df) > 0:

        action_accuracy = accuracy_score(
            action_df["expected_action"],
            action_df["predicted_action"]
        )

        action_f1 = f1_score(
            action_df["expected_action"],
            action_df["predicted_action"],
            pos_label="ESCALATE",
            zero_division=0
        )

        print(
            f"Action Accuracy : {action_accuracy:.4f}"
        )

        print(
            f"Escalation F1   : {action_f1:.4f}"
        )

        print("\nPredicted Action Distribution:")

        print(
            action_df["predicted_action"]
            .value_counts()
        )

        print("\nExpected Action Distribution:")

        print(
            action_df["expected_action"]
            .value_counts()
        )

        print("\nAction Classification Report:\n")

        print(
            classification_report(
                action_df["expected_action"],
                action_df["predicted_action"],
                zero_division=0
            )
        )

    else:

        print(
            "No valid expected_action labels found."
        )

else:

    print(
        "WARNING: expected_action column not found."
    )


# ============================================================
# 8. CONFIDENCE ANALYSIS
# ============================================================

print("\n================================")
print("CONFIDENCE ANALYSIS")
print("================================")

print(
    golden_df["confidence"].describe()
)


print("\nConfidence by predicted action:")

print(
    golden_df.groupby(
        "predicted_action"
    )["confidence"].agg(
        [
            "count",
            "mean",
            "min",
            "max"
        ]
    )
)


# ============================================================
# 9. CLASSIFICATION FAILURE ANALYSIS
# ============================================================

errors = golden_df[
    golden_df["golden_intent"]
    != golden_df["predicted_intent"]
].copy()


print("\n================================")
print("TOP CLASSIFICATION ERRORS")
print("================================")

print(
    f"Total errors: {len(errors)}"
)


if len(errors) > 0:

    error_pairs = (
        errors
        .groupby(
            [
                "golden_intent",
                "predicted_intent"
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print("\nMost common error pairs:\n")

    print(
        error_pairs.head(10)
    )


# ============================================================
# 10. LOW-CONFIDENCE PREDICTIONS
# ============================================================

print("\n================================")
print("LOW-CONFIDENCE PREDICTIONS")
print("================================")

low_confidence = golden_df[
    golden_df["confidence"] < THRESHOLD
].copy()

print(
    f"Low-confidence examples: "
    f"{len(low_confidence)}"
)


if len(low_confidence) > 0:

    print("\nExamples:\n")

    print(
        low_confidence[
            [
                "clean_text",
                "golden_intent",
                "predicted_intent",
                "confidence",
                "predicted_action",
                "predicted_escalation_reason"
            ]
        ]
        .sort_values("confidence")
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# 11. SAVE COMPLETE EVALUATION
# ============================================================

golden_df.to_csv(
    "evaluation/agent_evaluation.csv",
    index=False
)

print("\n================================")
print("EVALUATION COMPLETE")
print("================================")

print(
    "Complete evaluation saved to:"
)

print(
    "evaluation/agent_evaluation.csv"
)