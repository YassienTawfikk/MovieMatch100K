import src
from src.data_setup import download_dataset, data_preprocessing, split_data
from src.user_cf import recommend_for_user
from src.evaluation import evaluate_recommendation
import pandas as pd


def main():
    print("Downloading MovieLens 100K from KaggleHub...")
    download_dataset()

    print("Preprocessing Dataset...")
    data_preprocessing()

    print("Splitting Dataset...")
    split_data()

    # # Choose a user to recommend for
    # target_user_id = 13
    # k_movies = 5
    # print(f"Generating recommendations for user {target_user_id}...")
    #
    # # Get recommendations + matrices once
    # top_recs, user_item_matrix, similarity_matrix = recommend_for_user(
    #     user_id=target_user_id,
    #     k=k_movies,
    #     top_n_neighbors=50
    # )
    #
    # print(f"Top 5 recommendations for user {target_user_id}:")
    # for item_id, score in top_recs:
    #     print(f"Movie ID {item_id} → Predicted Rating: {score:.2f}")
    #
    # print("Evaluating model (This may take 3–5 mins)...")
    #
    # evaluate_recommendation(
    #     user_item_matrix=user_item_matrix,
    #     similarity_matrix=similarity_matrix,
    #     k=k_movies,
    #     top_n_neighbors=50
    # )


if __name__ == "__main__":
    main()
