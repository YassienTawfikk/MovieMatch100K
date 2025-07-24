import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_item_similarity_matrix(ratings_df):
    """
    Build item-item similarity matrix using cosine similarity on user-item ratings.

    Returns:
        user_item_matrix: user-item rating matrix
        item_similarity_matrix: item-item cosine similarity matrix
    """
    # Create user-item matrix (users as rows, items as columns)
    user_item_matrix = ratings_df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

    # Transpose: items as rows → compute cosine similarity between items
    item_sim = cosine_similarity(user_item_matrix.T)

    item_similarity_matrix = pd.DataFrame(
        item_sim,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    return user_item_matrix, item_similarity_matrix


def recommend_for_user_itemcf(user_id, ratings_df, user_item_matrix, item_similarity_matrix, k=5):
    """
    Recommend top-K items for a user using item-based collaborative filtering.

    Parameters:
        user_id: target user ID
        ratings_df: full ratings DataFrame
        user_item_matrix: matrix of user-item ratings
        item_similarity_matrix: precomputed item-item similarity matrix
        k: number of recommendations to return

    Returns:
        List of (item_id, predicted_score) tuples
    """

    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]

    # Use only items the user rated positively (e.g., 4 or 5)
    rated_items = user_ratings[user_ratings >= 4]

    scores = {}
    sim_sums = {}

    for item_id, rating in rated_items.items():
        similar_items = item_similarity_matrix[item_id]

        # Ignore items already rated by the user
        for sim_item_id, sim_score in similar_items.items():
            if sim_item_id in rated_items.index:
                continue

            if sim_item_id not in scores:
                scores[sim_item_id] = 0
                sim_sums[sim_item_id] = 0

            scores[sim_item_id] += sim_score * rating
            sim_sums[sim_item_id] += sim_score

    # Normalize scores by similarity sum (optional, for fairness)
    for item in scores:
        if sim_sums[item] > 0:
            scores[item] /= sim_sums[item]

    # Sort and get top-K
    top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    return top_k
