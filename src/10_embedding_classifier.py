import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# 1. LOAD DATA
# ==============================

print("Loading datasets...")

train_df = pd.read_csv("data/processed/apple_training.csv")
golden_df = pd.read_csv("evaluation/golden_set.csv")

train_df["clean_text"] = train_df["clean_text"].fillna("")
golden_df["clean_text"] = golden_df["clean_text"].fillna("")

print(f"Training examples : {len(train_df):,}")
print(f"Golden test       : {len(golden_df):,}")


# ==============================
# 2. LOAD SENTENCE TRANSFORMER
# ==============================

print("\nLoading sentence transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")


# ==============================
# 3. CREATE INTENT PROTOTYPES
# ==============================
# Instead of comparing every test tweet with all 38k
# training examples, create one semantic prototype
# for each intent.

intents = sorted(train_df["intent"].unique())

print(f"\nCreating prototypes for {len(intents)} intents...")

intent_embeddings = []

for intent in intents:
    examples = train_df[
        train_df["intent"] == intent
    ]["clean_text"].tolist()

    # Use up to 500 examples per intent
    examples = examples[:500]

    embeddings = model.encode(
        examples,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    prototype = np.mean(embeddings, axis=0)

    # Normalize prototype
    prototype = prototype / np.linalg.norm(prototype)

    intent_embeddings.append(prototype)

intent_embeddings = np.vstack(intent_embeddings)

print("Intent prototypes created.")


# ==============================
# 4. EMBED GOLDEN TEST SET
# ==============================

print("\nEmbedding Golden Set...")

test_embeddings = model.encode(
    golden_df["clean_text"].tolist(),
    show_progress_bar=True,
    normalize_embeddings=True
)


# ==============================
# 5. PREDICT INTENT
# ==============================

similarities = cosine_similarity(
    test_embeddings,
    intent_embeddings
)

best_indices = np.argmax(similarities, axis=1)

predicted_intents = [
    intents[i]
    for i in best_indices
]

confidence_scores = np.max(similarities, axis=1)

golden_df["predicted_intent"] = predicted_intents
golden_df["confidence"] = confidence_scores


# ==============================
# 6. EVALUATION
# ==============================

y_true = golden_df["golden_intent"]
y_pred = golden_df["predicted_intent"]

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\n================================")
print("MAIN AI — EMBEDDING CLASSIFIER")
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


# ==============================
# 7. SAVE RESULTS
# ==============================

output_columns = [
    "tweet_id",
    "clean_text",
    "golden_intent",
    "predicted_intent",
    "confidence"
]

# Keep only columns that actually exist
output_columns = [
    col for col in output_columns
    if col in golden_df.columns
]

golden_df[output_columns].to_csv(
    "evaluation/embedding_results.csv",
    index=False
)

print("\nResults saved to:")
print("evaluation/embedding_results.csv")


# ==============================
# 8. CONFIDENCE SUMMARY
# ==============================

print("\nConfidence Summary:")

print(
    golden_df["confidence"].describe()
)

print("\nLowest-confidence examples:")

print(
    golden_df[
        [
            "clean_text",
            "golden_intent",
            "predicted_intent",
            "confidence"
        ]
    ]
    .sort_values("confidence")
    .head(10)
    .to_string(index=False)
)