import pandas as pd
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "apple_customers.csv"

df = pd.read_csv(INPUT_FILE)

print("AppleSupport customer messages:", len(df))

# -----------------------------
# Basic text cleaning
# -----------------------------
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["clean_text"] = df["text"].apply(clean_text)

# Remove extremely short messages
df = df[df["clean_text"].str.len() >= 15].copy()

print("Usable messages:", len(df))

# -----------------------------
# Keyword-based topic exploration
# -----------------------------
topics = {
    "ios_update": [
        "ios", "update", "updating", "upgrade"
    ],
    "battery": [
        "battery", "charging", "charge"
    ],
    "iphone_device": [
        "iphone", "ipad", "ipod", "device"
    ],
    "app_issue": [
        "app", "apps", "application", "crash", "crashing"
    ],
    "apple_id_icloud": [
        "apple id", "icloud", "login", "sign in", "password"
    ],
    "app_store": [
        "app store", "itunes", "download", "purchase"
    ],
    "music": [
        "apple music", "music", "itunes music"
    ],
    "connectivity": [
        "wifi", "wi-fi", "bluetooth", "network", "internet",
        "cellular", "signal"
    ],
    "screen_display": [
        "screen", "display", "keyboard", "touch"
    ],
    "payment_billing": [
        "charge", "charged", "payment", "billing", "refund",
        "money", "subscription"
    ],
    "performance": [
        "slow", "slower", "freeze", "freezing", "lag", "lagging"
    ],
    "order_product": [
        "order", "ordered", "shipping", "delivery", "preorder"
    ]
}

print("\n========== TOPIC COUNTS ==========")

topic_counts = {}

for topic, keywords in topics.items():
    pattern = "|".join(re.escape(k) for k in keywords)

    count = df["clean_text"].str.contains(
        pattern,
        case=False,
        regex=True,
        na=False
    ).sum()

    topic_counts[topic] = count

for topic, count in sorted(
    topic_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{topic:20s} {count:8,}")

# -----------------------------
# Representative examples
# -----------------------------
print("\n========== EXAMPLES ==========")

for topic, keywords in topics.items():

    pattern = "|".join(re.escape(k) for k in keywords)

    matches = df[
        df["clean_text"].str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ]

    if len(matches) == 0:
        continue

    print(f"\n--- {topic.upper()} ---")

    examples = matches["clean_text"].sample(
        min(5, len(matches)),
        random_state=42
    )

    for i, text in enumerate(examples, 1):
        print(f"{i}. {text[:300]}")

# -----------------------------
# Save topic counts
# -----------------------------
counts_df = pd.DataFrame(
    list(topic_counts.items()),
    columns=["topic", "count"]
).sort_values("count", ascending=False)

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "topic_counts.csv"

counts_df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)