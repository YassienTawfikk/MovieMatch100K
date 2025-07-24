import pandas as pd
from src.data_setup import download_dataset, data_preprocessing, split_data
from src.user_cf import run_user_cf_pipeline
from src.item_cf import run_item_cf_pipeline
from src.svd import run_svd_pipeline
from src.evaluation import (
    evaluate_recommendation_user_cf,
    evaluate_recommendation_item_cf,
    evaluate_recommendation_svd
)


def main(method="user", evaluate=True, k_movies=5):
    # Step 1: Prepare data
    print("🔽 Downloading MovieLens 100K from KaggleHub...")
    download_dataset()

    print("🧹 Preprocessing dataset...")
    data_preprocessing()

    print("✂️ Splitting dataset into train/test sets...")
    split_data()

    # Step 2: Setup
    target_user_id = 13
    top_n_neighbors = 50

    # Load movie titles for pretty printing
    movies_df = pd.read_csv("data/processed/movies.csv")
    id_to_title = dict(zip(movies_df["item_id"], movies_df["title"]))

    if method == "user":
        print(f"\n📡 Running User-Based CF for user {target_user_id}...")
        top_recs_user, user_item_matrix, user_similarity_matrix = run_user_cf_pipeline(
            user_id=target_user_id,
            k=k_movies,
            top_n_neighbors=top_n_neighbors
        )

        print(f"🎯 Top {k_movies} User-Based Recommendations for User {target_user_id}:")
        for item_id, score in top_recs_user:
            movie_title = id_to_title.get(item_id, f"Movie ID {item_id}")
            print(f"  → {movie_title} | Predicted Rating: {score:.2f}")

        if evaluate:
            print("\n📈 Evaluating User-Based CF Model...")
            evaluate_recommendation_user_cf(
                user_item_matrix=user_item_matrix,
                similarity_matrix=user_similarity_matrix,
                k=k_movies,
                top_n_neighbors=top_n_neighbors
            )

    elif method == "item":
        print(f"\n🎞️ Running Item-Based CF for user {target_user_id}...")
        train_df = pd.read_csv("data/curated/train.csv")
        top_recs_item, item_similarity_matrix = run_item_cf_pipeline(
            user_id=target_user_id,
            ratings_df=train_df,
            k=k_movies
        )

        print(f"🎯 Top {k_movies} Item-Based Recommendations for User {target_user_id}:")
        for item_id, score in top_recs_item:
            movie_title = id_to_title.get(item_id, f"Movie ID {item_id}")
            print(f"  → {movie_title} | Predicted Score: {score:.2f}")

        if evaluate:
            print("\n📈 Evaluating Item-Based CF Model...")
            user_item_matrix = train_df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
            evaluate_recommendation_item_cf(
                user_item_matrix=user_item_matrix,
                item_similarity_matrix=item_similarity_matrix,
                k=k_movies
            )

    elif method == "svd":
        print(f"\n🧠 Running SVD for user {target_user_id}...")
        top_recs_svd, user_factors, item_factors, user_item_matrix = run_svd_pipeline(
            user_id=target_user_id,
            k=k_movies
        )

        print(f"🎯 Top {k_movies} SVD-Based Recommendations for User {target_user_id}:")
        for item_id, score in top_recs_svd:
            movie_title = id_to_title.get(item_id, f"Movie ID {item_id}")
            print(f"  → {movie_title} | Predicted Rating: {score:.2f}")

        if evaluate:
            print("\n📈 Evaluating SVD Model...")
            evaluate_recommendation_svd(
                user_factors=user_factors,
                item_factors=item_factors,
                user_item_matrix=user_item_matrix,
                k=k_movies
            )

    else:
        print("❌ Invalid method! Please choose 'user', 'item', or 'svd'.")


if __name__ == "__main__":
    main(method="svd", evaluate=True, k_movies=5)
