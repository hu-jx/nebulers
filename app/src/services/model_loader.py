from pathlib import Path

import joblib

from services.mock_model import MockModel
from services.rail_corrugation_model import RailCorrugationModel


_MODEL_DIR = Path(__file__).resolve().parent.parent / "model_store"


# looks for either the key name or the older *_model filename
def _find(key):
    candidates = [
        _MODEL_DIR / f"{key}.joblib",
        _MODEL_DIR / f"{key}_model.joblib",
        _MODEL_DIR / f"{key}.pt",
        _MODEL_DIR / f"{key}_model.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


# loads the real rail wrapper while leaving the other subsystem flow unchanged
def load(subsystem):
    path = _find(subsystem.key)

    if path is None:
        if subsystem.key == "rail_corrugation":
            raise FileNotFoundError(
                "rail corrugation model not found in model_store"
            )
        return MockModel(subsystem)

    if path.suffix == ".joblib":
        try:
            loaded = joblib.load(path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load model: {path.name}"
            ) from exc

        if subsystem.key == "rail_corrugation":
            return RailCorrugationModel(loaded)

        return loaded

    try:
        import torch
        return torch.load(
            path,
            map_location="cpu",
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load model: {path.name}"
        ) from exc
