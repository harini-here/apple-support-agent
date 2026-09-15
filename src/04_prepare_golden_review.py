import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "golden_candidates.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "evaluation"
    / "golden_review.csv"
)

df = pd.read_csv(INPUT_FILE)

# Target: approximately 200 examples
# Keep the classes balanced.
per_intent = 16

review_parts = []

for intent, group in df.groupby("candidate_intent"):

    n = min(per_intent, len(group))

    sample = group.sample(
        n=n,
        random_state=42
    ).copy()

    review_parts.append(sample)

review = pd.concat(
    review_parts,
    ignore_index=True
)

# Shuffle
review = review.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Add human-review columns
review["golden_intent"] = ""
review["expected_action"] = ""
review["escalation_reason"] = ""

review = review[
    [
        "tweet_id",
        "clean_text",
        "candidate_intent",
        "golden_intent",
        "expected_action",
        "escalation_reason"
    ]
]

review.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Golden review file created.")
print(f"Total examples: {len(review)}")
print(f"Saved to: {OUTPUT_FILE}")

print("\nCandidate distribution:")
print(review["candidate_intent"].value_counts())