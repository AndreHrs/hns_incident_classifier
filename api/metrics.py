"""Leaderboard and run-summary helpers backed by MLflow."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _flag(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes"}


def _safe_col(runs: pd.DataFrame, col: str, default_dtype=float) -> pd.Series:
    return runs[col] if col in runs.columns else pd.Series([None] * len(runs), dtype=default_dtype)


def get_leaderboard(
    sort_by: str = "val_f1_macro",
    ascending: bool = False,
    model_type_filter: str | None = None,
    architecture_filter: str | None = None,
) -> pd.DataFrame:
    """Query MLflow for all finished training runs and return a leaderboard DataFrame."""
    import mlflow

    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="status = 'FINISHED'",
    )

    if runs.empty:
        raise FileNotFoundError("No finished MLflow runs found.")

    df = pd.DataFrame()
    df["run_id"] = runs["run_id"]
    df["model_path"] = runs["run_id"]

    df["timestamp"] = (
        pd.to_datetime(runs["start_time"], unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    df["model_type"] = _safe_col(runs, "params.model_type", str)
    df["energy_model"] = _safe_col(runs, "params.energy_model", str)

    # Map canonical sort-key names to MLflow metric names logged by train_loop.py
    _metric_map = {
        "val_f1_macro": "best_val_f1_macro",
        "val_f1_weighted": "best_val_f1_weighted",
        "val_accuracy": "best_val_accuracy",
        "val_precision_macro": "best_val_precision_macro",
        "val_recall_macro": "best_val_recall_macro",
        "test_f1_macro": "test_f1_macro",
        "test_f1_weighted": "test_f1_weighted",
        "test_accuracy": "test_accuracy",
        "training_time_sec": "training_time_sec",
    }
    for col, mlflow_key in _metric_map.items():
        df[col] = _safe_col(runs, f"metrics.{mlflow_key}")

    if model_type_filter is not None:
        mt = model_type_filter.strip().lower()
        if mt not in {"energy", "damage"}:
            raise ValueError("model_type_filter must be 'energy' or 'damage'")
        want_energy = mt == "energy"
        df = df[df["energy_model"].map(_flag) == want_energy]

    if architecture_filter:
        df = df[_matches_architecture_filter(df["model_type"], architecture_filter)]

    if sort_by in df.columns:
        df = df.sort_values(by=[sort_by], ascending=ascending, na_position="last")

    return df.reset_index(drop=True)


def _matches_architecture_filter(series: pd.Series, architecture_filter: str) -> pd.Series:
    needle = architecture_filter.strip().lower()
    s_norm = series.astype(str).str.strip().str.lower().str.replace(" ", "_")

    if needle == "tf_idf":
        return s_norm.eq("tf_idf")
    if needle == "bert":
        return s_norm.eq("bert")
    if needle == "looped_transformer":
        return s_norm.eq("looped_transformer")
    if needle == "bigru":
        return s_norm.eq("bigru") | s_norm.str.startswith("bigru_")
    return s_norm.eq(needle)


def get_model_details(run_id: str) -> dict[str, Any]:
    """Return run params, metrics, and tags for an MLflow run."""
    import mlflow

    run = mlflow.get_run(run_id)
    return {
        "run_id": run_id,
        "run_name": run.info.run_name,
        "config": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "best_metric_value": run.data.metrics.get("best_metric_value"),
        "tags": dict(run.data.tags),
        "status": run.info.status,
    }
