import hashlib
from pathlib import Path

ROOT_DIR = Path.cwd()
dataset_path = ROOT_DIR / "mutag-hetero"

with open(dataset_path / "pyg_dataset.pt", "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest())