import src
from src.data_setup import *
from src.user_cf import recommend_for_user
import pandas as pd


def main():
    print("Downloading MovieLens 100K from KaggleHub...")
    download_dataset()

    print("Preprocessing Dataset...")
    data_preprocessing()

    print("Splitting Dataset...")
    split_data()

    # Choose a user to recommend for
    target_user_id = 13
    print(f"Generating recommendations for user {target_user_id}...")

    top_recs = recommend_for_user(user_id=target_user_id, k=5, top_n_neighbors=50)

    print(f"Top 5 recommendations for user {target_user_id}:")
    for item_id, score in top_recs:
        print(f"Movie ID {item_id} → Predicted Rating: {score:.2f}")


if __name__ == "__main__":
    main()
