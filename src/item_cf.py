import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_item_similarity_matrix(ratings_df):
    """
    Build item-item similarity matrix using cosine similarity on user-item ratings.
    """
    user_item_matrix = ratings_df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

    item_sim = cosine_similarity(user_item_matrix.T)
    item_similarity_matrix = pd.DataFrame(
        item_sim,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    return user_item_matrix, item_similarity_matrix


def recommend_for_user_item_cf(user_id, ratings_df, user_item_matrix, item_similarity_matrix, k=5):
    """
    Recommend top-K items for a user using item-based collaborative filtering.
    """
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings >= 4]

    scores = {}
    sim_sums = {}

    for item_id, rating in rated_items.items():
        similar_items = item_similarity_matrix[item_id]
        for sim_item_id, sim_score in similar_items.items():
            if sim_item_id in rated_items.index:
                continue
            if sim_item_id not in scores:
                scores[sim_item_id] = 0
                sim_sums[sim_item_id] = 0
            scores[sim_item_id] += sim_score * rating
            sim_sums[sim_item_id] += sim_score

    for item in scores:
        if sim_sums[item] > 0:
            scores[item] /= sim_sums[item]

    top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return top_k


def run_item_cf_pipeline(user_id, ratings_df=None, k=5):
    """
    Run the full item-based CF pipeline: build matrices and get recommendations.
    Returns top-K recommendations and the item similarity matrix.
    """
    if ratings_df is None:
        ratings_df = pd.read_csv("data/curated/train.csv")  # train, not ratings!

    user_item_matrix, item_similarity_matrix = build_item_similarity_matrix(ratings_df)

    recommendations = recommend_for_user_item_cf(
        user_id=user_id,
        ratings_df=ratings_df,
        user_item_matrix=user_item_matrix,
        item_similarity_matrix=item_similarity_matrix,
        k=k
    )

    return recommendations, item_similarity_matrix
