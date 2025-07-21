import kagglehub
import os
import shutil


def download_dataset(dest_folder="../data/raw"):
    print("Downloading MovieLens 100K from KaggleHub...")
    dataset_path = kagglehub.dataset_download("prajitdatta/movielens-100k-dataset")

    # Go inside the actual dataset folder (usually /.../ml-100k)
    dataset_subfolder = os.path.join(dataset_path, "ml-100k")
    if not os.path.exists(dataset_subfolder):
        raise Exception(f"Expected 'ml-100k' folder not found in {dataset_path}")

    os.makedirs(dest_folder, exist_ok=True)

    # Copy each file inside ml-100k to dest_folder
    for filename in os.listdir(dataset_subfolder):
        src_file = os.path.join(dataset_subfolder, filename)
        dst_file = os.path.join(dest_folder, filename)
        shutil.copy2(src_file, dst_file)

    print(f"Dataset successfully copied to: {dest_folder}")
    return dest_folder


if __name__ == "__main__":
    download_dataset()
