import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import NearestNeighbors


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
# 2. LOAD MODEL
# ==============================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded.")


# ==============================
# 3. CREATE TRAINING EMBEDDINGS
# ==============================

print("\nEmbedding training data...")

train_embeddings = model.encode(
    train_df["clean_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Training embeddings created.")


# ==============================
# 4. CREATE TEST EMBEDDINGS
# ==============================

print("\nEmbedding Golden Set...")

test_embeddings = model.encode(
    golden_df["clean_text"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Test embeddings created.")


# ==============================
# 5. BUILD NEAREST NEIGHBOR INDEX
# ==============================

print("\nBuilding nearest-neighbor index...")

knn = NearestNeighbors(
    n_neighbors=7,
    metric="cosine",
    algorithm="brute"
)

knn.fit(train_embeddings)

distances, indices = knn.kneighbors(test_embeddings)

print("Nearest-neighbor search complete.")


# ==============================
# 6. PREDICT USING WEIGHTED VOTING
# ==============================

predictions = []
confidence_scores = []

train_intents = train_df["intent"].values

for row_distances, row_indices in zip(distances, indices):

    neighbor_intents = train_intents[row_indices]

    # Convert cosine distance to similarity
    similarities = 1 - row_distances

    # Weighted voting
    scores = {}

    for intent, similarity in zip(
        neighbor_intents,
        similarities
    ):
        scores[intent] = scores.get(intent, 0) + similarity

    # Highest weighted score
    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    total_score = sum(scores.values())

    confidence = best_score / total_score

    predictions.append(best_intent)
    confidence_scores.append(confidence)


golden_df["predicted_intent"] = predictions
golden_df["confidence"] = confidence_scores


# ==============================
# 7. EVALUATION
# ==============================

y_true = golden_df["golden_intent"]
y_pred = golden_df["predicted_intent"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

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
print("MAIN AI — EMBEDDING KNN")
print("================================")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ==============================
# 8. CLASSIFICATION REPORT
# ==============================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)


# ==============================
# 9. SAVE RESULTS
# ==============================

output_columns = [
    "tweet_id",
    "clean_text",
    "golden_intent",
    "predicted_intent",
    "confidence"
]

output_columns = [
    col for col in output_columns
    if col in golden_df.columns
]

golden_df[output_columns].to_csv(
    "evaluation/embedding_knn_results.csv",
    index=False
)

print("\nResults saved to:")
print("evaluation/embedding_knn_results.csv")


# ==============================
# 10. CONFIDENCE ANALYSIS
# ==============================

print("\nConfidence Summary:")

print(
    golden_df["confidence"].describe()
)


# ==============================
# 11. LOW CONFIDENCE EXAMPLES
# ==============================

print("\nLowest-confidence examples:\n")

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