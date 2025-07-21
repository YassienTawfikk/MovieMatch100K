from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
CURATED_DIR = Path("data/curated")


def load_ratings():
    return pd.read_csv(PROCESSED_DIR / "ratings.csv")


def load_movies():
    return pd.read_csv(PROCESSED_DIR / "movies.csv")


def load_users():
    return pd.read_csv(PROCESSED_DIR / "users.csv")


def load_genres():
    return pd.read_csv(PROCESSED_DIR / "genres.csv")


def load_occupations():
    return pd.read_csv(PROCESSED_DIR / "occupations.csv")


def load_train_test():
    train = pd.read_csv(CURATED_DIR / "train.csv")
    test = pd.read_csv(CURATED_DIR / "test.csv")
    return train, test
