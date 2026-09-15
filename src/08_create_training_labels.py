import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "apple_customers.csv"
)

GOLDEN_FILE = (
    BASE_DIR
    / "evaluation"
    / "golden_set.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "apple_training.csv"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading AppleSupport customer data...")

df = pd.read_csv(INPUT_FILE)

golden = pd.read_csv(GOLDEN_FILE)

print(f"Raw customer messages: {len(df):,}")


# --------------------------------------------------
# Clean text
# --------------------------------------------------

def clean_text(text):
    text = str(text)

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)


# --------------------------------------------------
# Intent labeling rules
# --------------------------------------------------

RULES = {

    "battery_charging": [
        "battery",
        "battery drain",
        "battery life",
        "charging",
        "charger",
        "overheating",
        "overheat",
        "won't charge",
        "wont charge",
        "not charging"
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "network",
        "no service",
        "cellular",
        "mobile data",
        "signal"
    ],

    "apple_id_icloud": [
        "apple id",
        "icloud",
        "i cloud",
        "forgot password",
        "password",
        "sign in",
        "can't login",
        "cant login",
        "can't sign in",
        "cant sign in"
    ],

    "app_store_itunes": [
        "app store",
        "itunes",
        "download apps",
        "downloading apps",
        "can't download",
        "cant download"
    ],

    "apple_music_media": [
        "apple music",
        "music app",
        "airpods",
        "podcast",
        "music"
    ],

    "billing_payment_purchase": [
        "payment",
        "billing",
        "charged",
        "refund",
        "subscription",
        "purchase",
        "money taken",
        "payment information"
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

    "screen_display_keyboard": [
        "screen",
        "display",
        "keyboard",
        "touch screen",
        "touch",
        "white screen",
        "cracked screen"
    ],

    "app_issues": [
        "app crash",
        "app crashes",
        "apps crash",
        "app crashing",
        "apps crashing",
        "app freezing",
        "app frozen",
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
        "frozen",
        "performance"
    ],

    "ios_update": [
        "ios update",
        "software update",
        "updated my iphone",
        "after updating",
        "ios 11",
        "ios 10",
        "ios 11.1",
        "ios 10.3"
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


# --------------------------------------------------
# Assign weak labels
# --------------------------------------------------

def assign_label(text):

    text = text.lower()

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

    if scores[best_intent] == 0:
        return "other"

    return best_intent


print("\nGenerating weak labels...")

df["intent"] = df["clean_text"].apply(
    assign_label
)


# --------------------------------------------------
# Remove Golden Set from training data
# --------------------------------------------------

golden_ids = set(
    golden["tweet_id"]
)

before = len(df)

df = df[
    ~df["tweet_id"].isin(golden_ids)
].copy()

print(
    f"Removed {before - len(df):,} Golden Set examples "
    "from training data."
)


# --------------------------------------------------
# Remove very short messages
# --------------------------------------------------

df = df[
    df["clean_text"].str.len() >= 20
].copy()


# --------------------------------------------------
# Remove "other" for now
# --------------------------------------------------

df = df[
    df["intent"] != "other"
].copy()


# --------------------------------------------------
# Keep only useful columns
# --------------------------------------------------

training = df[
    [
        "tweet_id",
        "clean_text",
        "intent"
    ]
].copy()


# --------------------------------------------------
# Balance the training data
# --------------------------------------------------

MAX_PER_INTENT = 5000

balanced_parts = []

for intent, group in training.groupby("intent"):

    if len(group) > MAX_PER_INTENT:

        group = group.sample(
            MAX_PER_INTENT,
            random_state=42
        )

    balanced_parts.append(group)


training = pd.concat(
    balanced_parts,
    ignore_index=True
)


# --------------------------------------------------
# Shuffle
# --------------------------------------------------

training = training.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# --------------------------------------------------
# Save
# --------------------------------------------------

training.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n================================")
print("TRAINING DATA CREATED")
print("================================")

print(
    f"Training examples: {len(training):,}"
)

print("\nIntent distribution:")

print(
    training["intent"].value_counts()
)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)