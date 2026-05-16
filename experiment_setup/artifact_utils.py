"""
    _artifact_path_from_result:
            Get the path to the artifacts pickle file from a training result dictionary.
    _update_artifact_pickle:
            Merge updates into the just-saved artifacts pickle file.
"""

import pickle
from typing import Any
from pathlib import Path
from datetime import datetime


# Helpers for updating runner artifact pickle files.
def _artifact_path_from_result(result: dict[str, Any]) -> Path:
    cfg = result["config"]
    return Path(cfg["save_dir"]) / f"{cfg['save_name']}_artifacts.pkl"


def _update_artifact_pickle(result: dict[str, Any], updates: dict[str, Any]) -> None:
    """Merge updates into the just-saved artifacts pickle."""
    path = _artifact_path_from_result(result)

    with open(path, "rb") as f:
        artifacts = pickle.load(f)

    artifacts.update(updates)
    artifacts.setdefault("retrain_history", [])
    artifacts["retrain_history"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **updates.get("retrain_metadata", {}),
        }
    )

    with open(path, "wb") as f:
        pickle.dump(artifacts, f)