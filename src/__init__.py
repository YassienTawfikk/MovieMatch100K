from pathlib import Path

# Cleaner and modern alternative to os
paths = [
    Path("data/raw"),
    Path("data/processed"),
    Path("outputs")
]

for path in paths:
    path.mkdir(parents=True, exist_ok=True)

print("Directories created.")
