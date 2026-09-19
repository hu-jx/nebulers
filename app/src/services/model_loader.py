from functools import lru_cache
from pathlib import Path

import joblib

from services.door_model import DoorModel
from services.rail_corrugation_model import RailCorrugationModel
from services.shm_model import SHMModel


_MODEL_DIR = Path(__file__).resolve().parent.parent / "model_store"

_MODEL_FILES = {
    "door": "door_model.joblib",
    "rail_corrugation": "rail_corrugation_model.joblib",
    "shm": "shm_model.joblib",
}


# loads each saved artifact once and wraps it with the right app interface
@lru_cache(maxsize=None)
def _load_model(key):
    if key not in _MODEL_FILES:
        raise ValueError(
            f"unsupported subsystem: {key}"
        )

    path = _MODEL_DIR / _MODEL_FILES[key]

    if not path.exists():
        raise FileNotFoundError(
            f"model not found: {path.name}"
        )

    try:
        loaded = joblib.load(
            path
        )
    except Exception as exc:
        raise RuntimeError(
            f"failed to load model: {path.name}"
        ) from exc

    if key == "door":
        return DoorModel(
            loaded
        )

    if key == "rail_corrugation":
        return RailCorrugationModel(
            loaded
        )

    if key == "shm":
        return SHMModel(
            loaded
        )

    raise ValueError(
        f"unsupported subsystem: {key}"
    )


# keeps the controller interface unchanged
def load(subsystem):
    return _load_model(
        subsystem.key
    )
