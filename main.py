import src
from src import data_setup as data_setup


def main():
    print("Downloading MovieLens 100K from KaggleHub...")
    data_setup.download_dataset()

    print("Preprocessing Dataset...")
    data_setup.data_preprocessing()


if __name__ == "__main__":
    main()
