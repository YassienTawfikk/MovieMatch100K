import kagglehub
from pathlib import Path
import shutil
import pandas as pd


def download_dataset(dest_folder=Path("data/raw")):
    # Download and get dataset path
    dataset_path = Path(kagglehub.dataset_download("prajitdatta/movielens-100k-dataset"))
    dataset_subfolder = dataset_path / "ml-100k"

    # Validate expected subfolder
    if not dataset_subfolder.exists():
        raise FileNotFoundError(f"Expected 'ml-100k' folder not found in {dataset_path}")

    # Ensure destination folder exists
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Copy files
    for file in dataset_subfolder.iterdir():
        if file.is_file():
            shutil.copy2(file, dest_folder / file.name)

    return dest_folder


def data_preprocessing():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_csv(
        raw_dir / "u.data", sep='\t', header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    ratings = ratings.drop(columns=["timestamp"])
    ratings.to_csv(processed_dir / "ratings.csv", index=False)

    movie_columns = [
        "item_id", "title", "release_date", "video_release_date", "IMDb_URL",
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies = pd.read_csv(
        raw_dir / "u.item", sep='|', encoding='latin-1', header=None, names=movie_columns
    )
    movies.to_csv(processed_dir / "movies.csv", index=False)

    users = pd.read_csv(
        raw_dir / "u.user", sep='|', header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"]
    )
    users.to_csv(processed_dir / "users.csv", index=False)

    occupations = pd.read_csv(raw_dir / "u.occupation", header=None, names=["occupation"])
    occupations.to_csv(processed_dir / "occupations.csv", index=False)

    genres = pd.read_csv(raw_dir / "u.genre", sep='|', header=None, names=["genre", "genre_id"])
    genres.to_csv(processed_dir / "genres.csv", index=False)


def split_data(test_ratio=0.2, min_ratings=5, random_state=42):
    def split_per_user(df, test_ratio=0.2, min_ratings=5):
        train_rows = []
        test_rows = []

        for user_id, group in df.groupby("user_id"):
            if len(group) < min_ratings:
                # Not enough ratings → all go to train
                train_rows.extend(group.to_dict("records"))
            else:
                test_count = max(1, int(len(group) * test_ratio))
                test_sample = group.sample(test_count, random_state=random_state)
                train_sample = group.drop(test_sample.index)

                test_rows.extend(test_sample.to_dict("records"))
                train_rows.extend(train_sample.to_dict("records"))

        train_df = pd.DataFrame(train_rows)
        test_df = pd.DataFrame(test_rows)
        return train_df, test_df

    # --- Paths ---
    processed_dir = Path("data/processed")
    curated_dir = Path("data/curated")
    curated_dir.mkdir(parents=True, exist_ok=True)

    # --- Load full dataset ---
    ratings = pd.read_csv(processed_dir / "ratings.csv")

    # --- Perform smart split ---
    train_df, test_df = split_per_user(ratings, test_ratio=test_ratio, min_ratings=min_ratings)

    # --- Save ---
    train_df.to_csv(curated_dir / "train.csv", index=False)
    test_df.to_csv(curated_dir / "test.csv", index=False)

    print("✅ Per-user stratified split complete.")
    return train_df, test_df
