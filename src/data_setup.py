import kagglehub
from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


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


def split_data(test_size=0.2, random_state=42):
    processed_dir = Path("data/processed")
    curated_dir = Path("data/curated")

    ratings = pd.read_csv(processed_dir / "ratings.csv")

    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)

    train.to_csv(curated_dir / "train.csv", index=False)
    test.to_csv(curated_dir / "test.csv", index=False)

    return train, test
