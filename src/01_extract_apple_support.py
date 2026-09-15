import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "twcs.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(RAW_FILE)

print(f"Total tweets: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# -----------------------------
# Extract AppleSupport messages
# -----------------------------
apple_agent = df[
    df["author_id"].astype(str).str.lower() == "applesupport"
].copy()

print(f"\nAppleSupport agent tweets: {len(apple_agent):,}")

# -----------------------------
# Find customer tweets answered
# -----------------------------
response_ids = (
    apple_agent["in_response_to_tweet_id"]
    .dropna()
    .astype("int64")
)

apple_customers = df[
    df["tweet_id"].isin(response_ids)
].copy()

# Keep inbound/customer messages
apple_customers = apple_customers[
    apple_customers["inbound"] == True
].copy()

print(f"AppleSupport customer tweets: {len(apple_customers):,}")

# -----------------------------
# Save datasets
# -----------------------------
agent_file = OUTPUT_DIR / "apple_agent.csv"
customer_file = OUTPUT_DIR / "apple_customers.csv"

apple_agent.to_csv(agent_file, index=False)
apple_customers.to_csv(customer_file, index=False)

print("\nSaved:")
print(agent_file)
print(customer_file)

# -----------------------------
# Basic statistics
# -----------------------------
print("\n--- AppleSupport Summary ---")
print(f"Agent tweets:    {len(apple_agent):,}")
print(f"Customer tweets: {len(apple_customers):,}")
print(
    f"Unique customers: "
    f"{apple_customers['author_id'].nunique():,}"
)

print("\nDone!")