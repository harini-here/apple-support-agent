import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "apple_customers.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_candidates.csv"
)

df = pd.read_csv(INPUT_FILE)

# -----------------------------
# Clean text
# -----------------------------

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)

# Remove very short messages
df = df[df["clean_text"].str.len() >= 20].copy()

# -----------------------------
# Intent keywords
# -----------------------------

intent_keywords = {
    "ios_update": [
        "ios update",
        "ios 11",
        "ios 10",
        "ios 11.1",
        "updated my iphone",
        "after updating",
        "software update"
    ],

    "battery_charging": [
        "battery",
        "battery life",
        "battery drain",
        "draining",
        "charging",
        "charger",
        "charge my iphone",
        "overheating"
    ],

    "device_hardware": [
        "iphone",
        "ipad",
        "device",
        "phone",
        "hardware",
        "broken",
        "malfunction"
    ],

    "app_issues": [
        "app crashes",
        "app crash",
        "apps crash",
        "app freezing",
        "app not working",
        "application"
    ],

    "performance": [
        "slow",
        "slower",
        "lag",
        "lagging",
        "freezes",
        "freezing",
        "freeze"
    ],

    "screen_display_keyboard": [
        "screen",
        "display",
        "touch",
        "keyboard",
        "white screen",
        "screen cracked"
    ],

    "connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "network",
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
        "downloading apps",
        "can't download"
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
        "charge",
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
    ]
}

# -----------------------------
# Find candidates
# -----------------------------

candidates = []

for intent, keywords in intent_keywords.items():

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    matches = df[
        df["clean_text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    # Sample up to 20 per intent
    n = min(20, len(matches))

    if n > 0:
        sample = matches.sample(
            n=n,
            random_state=42
        ).copy()

        sample["candidate_intent"] = intent

        candidates.append(
            sample[
                [
                    "tweet_id",
                    "author_id",
                    "clean_text",
                    "candidate_intent"
                ]
            ]
        )

# -----------------------------
# Combine
# -----------------------------

golden = pd.concat(
    candidates,
    ignore_index=True
)

# Remove duplicate tweets
golden = golden.drop_duplicates(
    subset="tweet_id"
)

# Shuffle
golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

golden.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Golden candidates created!")
print(f"Total candidates: {len(golden)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nCandidate distribution:")
print(
    golden["candidate_intent"]
    .value_counts()
)