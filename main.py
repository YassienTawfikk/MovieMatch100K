import src
from src.data_setup import download_dataset, data_preprocessing, split_data


def main():
    print("Downloading MovieLens 100K from KaggleHub...")
    download_dataset()

    print("Preprocessing Dataset...")
    data_preprocessing()

    print("Splitting Dataset...")
    split_data()


if __name__ == "__main__":
    main()
