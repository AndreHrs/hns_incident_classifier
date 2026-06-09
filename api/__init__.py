"""High-level orchestration helpers for training, exporting, inference, and reporting.

This module intentionally avoids eagerly importing inference/model-loading
dependencies. The Streamlit training page imports `api` before calling
`api.train(...)`; eager imports here can pull in BERT/Transformers modules even
for TF-IDF training.
"""

from typing import Any

from .train import train

__all__ = [
    "train",
    "load_model",
    "infer",
    "get_leaderboard",
    "get_model_details",
    "retrain",
]


def __getattr__(name: str) -> Any:
    """Lazily expose heavier API functions only when requested.

    Each branch pins the resolved function back into module globals so that
    Python's submodule-import side-effect (which would set api.<name> to the
    submodule object) is overwritten and subsequent accesses return the function.
    """
    _g = globals()

    if name == "load_model":
        from .loader import load_model
        _g["load_model"] = load_model
        return load_model

    if name == "infer":
        from .infer import infer
        _g["infer"] = infer
        return infer

    if name == "get_leaderboard":
        from .metrics import get_leaderboard
        _g["get_leaderboard"] = get_leaderboard
        return get_leaderboard

    if name == "get_model_details":
        from .metrics import get_model_details
        _g["get_model_details"] = get_model_details
        return get_model_details

    if name == "retrain":
        from .retrain import retrain
        _g["retrain"] = retrain
        return retrain

    raise AttributeError(f"module 'api' has no attribute {name!r}")