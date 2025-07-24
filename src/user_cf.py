import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_matrix(ratings):
    """Pivot ratings into user-item matrix, fill NaNs with 0."""
    matrix = ratings.pivot(index="user_id", columns="item_id", values="rating")
    return matrix.fillna(0)


def compute_user_similarity(user_item_matrix):
    """Compute cosine similarity between users."""
    similarity = cosine_similarity(user_item_matrix)
    return pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def get_top_k_recommendations(user_id, ratings, user_item_matrix, similarity_matrix, k=5, top_n_neighbors=50):
    """Predict ratings for unseen movies for a user using top-N similar users."""

    # Movies the user already rated
    rated_items = ratings[ratings["user_id"] == user_id]["item_id"].tolist()

    # Movies the user hasn't rated yet
    unseen_items = [item for item in user_item_matrix.columns if item not in rated_items]

    # Get similarity scores for the target user
    sim_scores = similarity_matrix.loc[user_id]

    # Select top-N similar users (excluding self)
    sim_scores = sim_scores.drop(user_id)
    top_users = sim_scores.sort_values(ascending=False).head(top_n_neighbors)

    predictions = {}

    for item in unseen_items:
        item_ratings = user_item_matrix.loc[top_users.index, item]
        valid_mask = item_ratings > 0

        if valid_mask.sum() == 0:
            continue  # No similar users rated this item

        sim_subset = top_users[valid_mask]
        rating_subset = item_ratings[valid_mask]

        # Weighted average
        predicted_rating = np.dot(sim_subset, rating_subset) / sim_subset.sum()
        predictions[item] = predicted_rating

    # Return top-K predicted items
    top_k = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]
    return top_k


def run_user_cf_pipeline(user_id, ratings_df=None, k=5, top_n_neighbors=50):
    """
    Full pipeline: build matrix → similarity → recommend for one user.

    Parameters:
        user_id: target user ID (int)
        ratings_df: full ratings DataFrame (train)
        k: number of items to recommend
        top_n_neighbors: number of similar users to consider

    Returns:
        List of (item_id, predicted_rating) tuples
    """

    if ratings_df is None:
        ratings_df = pd.read_csv("data/curated/train.csv")

    user_item_matrix = build_user_item_matrix(ratings_df)
    similarity_matrix = compute_user_similarity(user_item_matrix)

    if user_id not in user_item_matrix.index:
        raise ValueError(f"User ID {user_id} not found in training data.")

    top_k = get_top_k_recommendations(
        user_id=user_id,
        ratings=ratings_df,
        user_item_matrix=user_item_matrix,
        similarity_matrix=similarity_matrix,
        k=k,
        top_n_neighbors=top_n_neighbors
    )
    return top_k, user_item_matrix, similarity_matrix
