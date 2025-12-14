from pathlib import Path

def repo_root() -> Path:
    # driftrpl/io.py -> repo root is 1 level up
    return Path(__file__).resolve().parents[1]

def data_dir() -> Path:
    return repo_root() / "data"
