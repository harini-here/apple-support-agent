import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_FILE = (
    BASE_DIR / "data" / "processed" / "golden_candidates.csv"
)

REVIEW_FILE = (
    BASE_DIR / "evaluation" / "golden_review.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "evaluation" / "golden_set.csv"
)

# Load the 192 already selected examples
review = pd.read_csv(REVIEW_FILE)

# Load the complete candidate pool
candidates = pd.read_csv(CANDIDATE_FILE)

# Find candidates not already used
remaining = candidates[
    ~candidates["tweet_id"].isin(review["tweet_id"])
].copy()

# Add 8 additional real examples
extra = (
    remaining
    .sample(n=8, random_state=123)
    .copy()
)

# Combine
golden = pd.concat(
    [review, extra],
    ignore_index=True
)

# Initial labels come from the candidate-generation process.
# These must be reviewed before being treated as ground truth.
golden["golden_intent"] = golden["candidate_intent"]

# Mark verification status explicitly.
golden["review_status"] = "REQUIRES_REVIEW"

# Empty fields for final human verification.
golden["expected_action"] = ""
golden["escalation_reason"] = ""

# Shuffle
golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

golden.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Golden Set candidate file created.")
print(f"Total examples: {len(golden)}")
print(f"Needs review: {(golden['review_status'] == 'REQUIRES_REVIEW').sum()}")

print("\nIntent distribution:")
print(golden["golden_intent"].value_counts())

print(f"\nSaved to: {OUTPUT_FILE}")