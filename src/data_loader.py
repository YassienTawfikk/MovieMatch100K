import kagglehub
import os


def download_dataset(dest_folder="data"):
    print("Downloading MovieLens 100K from KaggleHub...")
    dataset_path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")

    # Optional: Move to your project folder
    os.makedirs(dest_folder, exist_ok=True)
    print(f"Dataset downloaded and cached at: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    download_dataset()
