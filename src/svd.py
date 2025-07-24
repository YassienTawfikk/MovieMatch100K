import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD


def build_svd_model(ratings_df, n_components=20):
    """
    Builds the SVD model using TruncatedSVD from sklearn.

    Returns:
        user_factors: matrix of shape (n_users, n_components)
        item_factors: matrix of shape (n_components, n_items)
        user_item_matrix: the pivot matrix used
    """
    user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_

    return user_factors, item_factors, user_item_matrix


def recommend_for_user_svd(user_id, user_factors, item_factors, user_item_matrix, k=5):
    """
    Recommend top-K items for a given user using the SVD latent factors.

    Returns:
        List of (item_id, predicted_rating)
    """
    if user_id not in user_item_matrix.index:
        return []

    user_idx = user_item_matrix.index.get_loc(user_id)
    predicted_ratings = np.dot(user_factors[user_idx, :], item_factors)

    rated_items = user_item_matrix.columns[user_item_matrix.loc[user_id] > 0]
    unseen_items = [item for item in user_item_matrix.columns if item not in rated_items]

    recommendations = [(item_id, predicted_ratings[i]) for i, item_id in enumerate(user_item_matrix.columns) if
                       item_id in unseen_items]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:k]

    return recommendations


def run_svd_pipeline(user_id, ratings_df=None, k=5, n_components=20):
    if ratings_df is None:
        ratings_df = pd.read_csv("data/curated/train.csv")

    user_factors, item_factors, user_item_matrix = build_svd_model(ratings_df, n_components=n_components)

    top_k = recommend_for_user_svd(
        user_id=user_id,
        user_factors=user_factors,
        item_factors=item_factors,
        user_item_matrix=user_item_matrix,
        k=k
    )

    return top_k, user_factors, item_factors, user_item_matrix
