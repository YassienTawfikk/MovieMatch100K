from pathlib import Path

# Base directory relative to the current script's file location
base_dir = Path(__file__).resolve().parent

# Define subdirectories relative to the script's location
paths = [
    base_dir / "data" / "raw",
    base_dir / "data" / "processed",
    base_dir / "data" / "curated",
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

print("Directories are created or already exist.")