import pandas as pd
import re
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "evaluation" / "golden_set.csv"
OUTPUT_FILE = BASE_DIR / "evaluation" / "baseline_rules_results.csv"


# --------------------------------------------------
# Load Golden Set
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Golden Set size: {len(df)}")


# --------------------------------------------------
# Rule-based classifier
# --------------------------------------------------

RULES = {
    "battery_charging": [
        "battery",
        "charging",
        "charger",
        "overheating",
        "battery drain",
        "battery life",
        "won't charge",
        "wont charge"
    ],

    "screen_display_keyboard": [
        "screen",
        "display",
        "keyboard",
        "touch",
        "white screen",
        "cracked screen"
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "network",
        "no service",
        "cellular",
        "signal"
    ],

    "apple_id_icloud": [
        "apple id",
        "icloud",
        "i cloud",
        "password",
        "sign in",
        "login"
    ],

    "app_store_itunes": [
        "app store",
        "itunes",
        "download apps",
        "can't download",
        "cant download"
    ],

    "apple_music_media": [
        "apple music",
        "music app",
        "music",
        "airpods",
        "podcast"
    ],

    "billing_payment_purchase": [
        "payment",
        "billing",
        "charged",
        "refund",
        "subscription",
        "purchase"
    ],

    "orders_products_warranty": [
        "order",
        "ordered",
        "pre-order",
        "preorder",
        "shipping",
        "delivery",
        "warranty",
        "replacement"
    ],

    "app_issues": [
        "app crash",
        "app crashes",
        "apps crash",
        "app crashing",
        "app freezing",
        "app not working",
        "apps not working"
    ],

    "performance": [
        "slow",
        "slower",
        "lag",
        "lagging",
        "freeze",
        "freezing",
        "frozen"
    ],

    "ios_update": [
        "ios update",
        "software update",
        "updated my iphone",
        "after updating",
        "ios 11",
        "ios 10"
    ],

    "device_hardware": [
        "iphone",
        "ipad",
        "ipod",
        "device",
        "hardware",
        "malfunction",
        "broken"
    ]
}


def predict_intent(text):

    text = str(text).lower()

    scores = {}

    for intent, keywords in RULES.items():

        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(
        scores,
        key=scores.get
    )

    # No matching rule
    if scores[best_intent] == 0:
        return "other"

    return best_intent


# --------------------------------------------------
# Run predictions
# --------------------------------------------------

df["predicted_intent"] = df["clean_text"].apply(
    predict_intent
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

y_true = df["golden_intent"]
y_pred = df["predicted_intent"]

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

print("\n================================")
print("BASELINE #1 — RULE CLASSIFIER")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

labels = sorted(
    set(y_true) | set(y_pred)
)

cm = confusion_matrix(
    y_true,
    y_pred,
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

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nResults saved to:")
print(OUTPUT_FILE)