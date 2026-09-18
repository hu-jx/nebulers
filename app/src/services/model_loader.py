from pathlib import Path
import joblib
from services.mock_model import MockModel
_MODEL_DIR = Path(__file__).resolve().parent.parent / "model_store"

def _find(key):
    for ext in (".joblib", ".pt"):
        path = _MODEL_DIR / f"{key}{ext}"
        if path.exists():
            return path
    return None

def load(subsystem):
    path = _find(subsystem.key)
    if path is None:
        return MockModel(subsystem)
    try:
        if path.suffix == ".joblib":
            return joblib.load(path)
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        return MockModel(subsystem)