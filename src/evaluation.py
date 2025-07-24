import pandas as pd
import numpy as np

from src.item_cf import recommend_for_user_item_cf
from src.user_cf import get_top_k_recommendations
from src.svd import recommend_for_user_svd


# =====================================================================
# ============================== User CF ==============================
# =====================================================================

def precision_at_k_user_cf(actual_items, predicted_items, k=5):
    """
    Compute Precision@K for a single user.
    actual_items: set of true items the user interacted with in test set
    predicted_items: list of recommended item_ids
    """
    if not actual_items:
        return np.nan  # Skip users with no actual items

    hits = [item for item in predicted_items[:k] if item in actual_items]
    return len(hits) / k


def evaluate_precision_at_k_user_cf(test_df, train_df, user_item_matrix, similarity_matrix, k=5):
    """
    Evaluate average Precision@K across all users in the test set.
    """
    precisions = []

    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        # Get actual movies the user rated in test set
        actual_items = set(test_df[test_df["user_id"] == user_id]["item_id"].tolist())

        # Skip users who don’t exist in training
        if user_id not in user_item_matrix.index:
            continue

        # Get recommendations
        top_k_recs = get_top_k_recommendations(
            user_id=user_id,
            ratings=train_df,
            user_item_matrix=user_item_matrix,
            similarity_matrix=similarity_matrix,
            k=k
        )

        predicted_items = [item for item, _ in top_k_recs]
        prec = precision_at_k_user_cf(actual_items, predicted_items, k=k)

        if not np.isnan(prec):
            precisions.append(prec)

    return np.mean(precisions) if precisions else 0.0


def evaluate_recall_at_k_user_cf(test_df, train_df, user_item_matrix, similarity_matrix, k=5, top_n_neighbors=50):
    """
    Compute Recall@K for all users in the test set.
    """

    user_ids = test_df["user_id"].unique()
    recalls = []

    for user_id in user_ids:
        if user_id not in user_item_matrix.index:
            continue  # skip cold-start users

        # Ground truth: what this user actually rated in test set
        relevant_items = test_df[test_df["user_id"] == user_id]["item_id"].tolist()

        # Generate top-K predictions
        top_k_predicted = get_top_k_recommendations(
            user_id=user_id,
            ratings=train_df,
            user_item_matrix=user_item_matrix,
            similarity_matrix=similarity_matrix,
            k=k,
            top_n_neighbors=top_n_neighbors
        )

        recommended_items = [item for item, _ in top_k_predicted]

        # Compute recall
        hits = len(set(recommended_items) & set(relevant_items))
        possible = len(relevant_items)
        if possible > 0:
            recall = hits / possible
            recalls.append(recall)

    return sum(recalls) / len(recalls) if recalls else 0.0


def evaluate_recommendation_user_cf(test_df=None, train_df=None, user_item_matrix=None, similarity_matrix=None,
                                    k=5, top_n_neighbors=50):
    if test_df is None:
        test_df = pd.read_csv("data/curated/test.csv")
    if train_df is None:
        train_df = pd.read_csv("data/curated/train.csv")

    precision = evaluate_precision_at_k_user_cf(test_df=test_df, train_df=train_df, user_item_matrix=user_item_matrix,
                                                similarity_matrix=similarity_matrix, k=k)

    print(f"Average Precision@{k}: {precision * 100:.4f}%")

    recall = evaluate_recall_at_k_user_cf(test_df=test_df, train_df=train_df, user_item_matrix=user_item_matrix,
                                          similarity_matrix=similarity_matrix, k=k, top_n_neighbors=50)

    print(f"Average Recall@{k}: {recall * 100:.4f}%")


# =====================================================================
# ============================== Item CF ==============================
# =====================================================================

def evaluate_precision_item_cf(test_df, train_df, user_item_matrix, item_similarity_matrix, k=5):
    precisions = []
    users = test_df["user_id"].unique()

    for user_id in users:
        if user_id not in user_item_matrix.index:
            continue

        actual_items = set(test_df[test_df["user_id"] == user_id]["item_id"])

        top_k = recommend_for_user_item_cf(
            user_id=user_id,
            ratings_df=train_df,
            user_item_matrix=user_item_matrix,
            item_similarity_matrix=item_similarity_matrix,
            k=k
        )

        predicted_items = [item for item, _ in top_k]
        hits = [item for item in predicted_items if item in actual_items]
        precision = len(hits) / k if k else 0
        precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def evaluate_recall_item_cf(test_df, train_df, user_item_matrix, item_similarity_matrix, k=5):
    recalls = []
    users = test_df["user_id"].unique()

    for user_id in users:
        if user_id not in user_item_matrix.index:
            continue

        actual_items = set(test_df[test_df["user_id"] == user_id]["item_id"])

        top_k = recommend_for_user_item_cf(
            user_id=user_id,
            ratings_df=train_df,
            user_item_matrix=user_item_matrix,
            item_similarity_matrix=item_similarity_matrix,
            k=k
        )

        predicted_items = [item for item, _ in top_k]
        hits = len(set(predicted_items) & actual_items)
        recall = hits / len(actual_items) if actual_items else 0
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def evaluate_recommendation_item_cf(test_df=None, train_df=None, user_item_matrix=None, item_similarity_matrix=None,
                                    k=5):
    """
    Wrapper to evaluate both Precision@K and Recall@K for item-based CF.
    """
    if test_df is None:
        test_df = pd.read_csv("data/curated/test.csv")
    if train_df is None:
        train_df = pd.read_csv("data/curated/train.csv")

    precision = evaluate_precision_item_cf(test_df, train_df, user_item_matrix, item_similarity_matrix, k)
    print(f"Average Precision@{k}: {precision * 100:.4f}%")

    recall = evaluate_recall_item_cf(test_df, train_df, user_item_matrix, item_similarity_matrix, k)
    print(f"Average Recall@{k}: {recall * 100:.4f}%")


# =====================================================================
# ================================ SVD ================================
# =====================================================================

def evaluate_precision_at_k_svd(test_df, user_factors, item_factors, user_item_matrix, k=5):
    precisions = []
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        if user_id not in user_item_matrix.index:
            continue

        actual_items = set(test_df[test_df["user_id"] == user_id]["item_id"])

        # Generate recommendations
        top_k = recommend_for_user_svd(
            user_id=user_id,
            user_factors=user_factors,
            item_factors=item_factors,
            user_item_matrix=user_item_matrix,
            k=k
        )
        predicted_items = [item for item, _ in top_k]

        hits = [item for item in predicted_items if item in actual_items]
        precision = len(hits) / k if k else 0
        precisions.append(precision)

    return np.mean(precisions) if precisions else 0.0


def evaluate_recall_at_k_svd(test_df, user_factors, item_factors, user_item_matrix, k=5):
    recalls = []
    test_users = test_df["user_id"].unique()

    for user_id in test_users:
        if user_id not in user_item_matrix.index:
            continue

        actual_items = set(test_df[test_df["user_id"] == user_id]["item_id"])

        # Generate recommendations
        top_k = recommend_for_user_svd(
            user_id=user_id,
            user_factors=user_factors,
            item_factors=item_factors,
            user_item_matrix=user_item_matrix,
            k=k
        )
        predicted_items = [item for item, _ in top_k]

        hits = len(set(predicted_items) & actual_items)
        recall = hits / len(actual_items) if actual_items else 0
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def evaluate_recommendation_svd(test_df=None, train_df=None, user_factors=None, item_factors=None,
                                user_item_matrix=None, k=5):
    """
    Full evaluation pipeline for SVD: Precision@K and Recall@K
    """
    if test_df is None:
        test_df = pd.read_csv("data/curated/test.csv")
    if train_df is None:
        train_df = pd.read_csv("data/curated/train.csv")

    precision = evaluate_precision_at_k_svd(
        test_df=test_df,
        user_factors=user_factors,
        item_factors=item_factors,
        user_item_matrix=user_item_matrix,
        k=k
    )
    print(f"Average Precision@{k}: {precision * 100:.4f}%")

    recall = evaluate_recall_at_k_svd(
        test_df=test_df,
        user_factors=user_factors,
        item_factors=item_factors,
        user_item_matrix=user_item_matrix,
        k=k
    )
    print(f"Average Recall@{k}: {recall * 100:.4f}%")
