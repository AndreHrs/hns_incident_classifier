"""Streamlit entry point — multi-page app with shared helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import mlflow
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


class ModelEntry(NamedTuple):
    """Lightweight descriptor for a trained MLflow run."""

    label: str
    run_id: str


def list_trained_models(task_filter: str | None = None) -> list[ModelEntry]:
    """Query MLflow for all finished runs and return labelled model entries.

    task_filter: "energy" or "damage" — matched against the logged energy_model param.
    """
    import mlflow

    try:
        runs = mlflow.search_runs(
            search_all_experiments=True,
            filter_string="status = 'FINISHED'",
        )
    except Exception:
        return []

    if runs.empty:
        return []

    entries: list[ModelEntry] = []
    for _, row in runs.iterrows():
        model_type = row.get("params.model_type") or "unknown"
        energy_str = str(row.get("params.energy_model", "")).strip().lower()
        is_energy = energy_str in {"true", "1", "yes"}

        if task_filter == "energy" and not is_energy:
            continue
        if task_filter == "damage" and is_energy:
            continue

        run_id = row["run_id"]
        run_name = row.get("tags.mlflow.runName") or run_id[:8]
        best_val_f1 = row.get("metrics.best_val_f1_macro")

        if best_val_f1 is not None and not pd.isna(best_val_f1):
            label = f"{run_name} — {model_type} [val_f1_macro={float(best_val_f1):.4f}]"
        else:
            label = f"{run_name} — {model_type}"

        entries.append(ModelEntry(label=label, run_id=run_id))

    return entries


def save_uploaded_file(uploaded_file) -> Path:
    """Persist a Streamlit UploadedFile to a temp path and return it."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp") as f:
        f.write(uploaded_file.getbuffer())
        return Path(f.name)


DOCS_INDEX = Path(__file__).resolve().parents[1] / "docs" / "build" / "html" / "index.html"


def main():
    """Run the Streamlit multi-page app."""
    st.set_page_config(page_title="Incident Classifier", layout="wide")

    train_pg = st.Page("pages/train_page.py", title="Train", icon="🏋️")
    infer_pg = st.Page("pages/inference_page.py", title="Inference", icon="🔍")
    metrics_pg = st.Page("pages/metrics_page.py", title="Metrics", icon="📊")
    review_pg = st.Page("pages/human_review_page.py", title="Human Review", icon="👁️")
    report_pg = st.Page("pages/report_page.py", title="Report", icon="📄")
    settings_pg = st.Page("pages/settings_page.py", title="Settings", icon="⚙️")
    retraining_pg = st.Page("pages/retraining_page.py", title="Retraining", icon="🔄")
    about_pg = st.Page("pages/about_page.py", title="About", icon="ℹ️")

    pg = st.navigation([train_pg, infer_pg, metrics_pg, review_pg, report_pg, settings_pg, retraining_pg, about_pg])

    with st.sidebar:
        if DOCS_INDEX.exists():
            docs_url = DOCS_INDEX.as_uri()
        else:
            docs_url = None
        if docs_url:
            st.markdown(f"[📚 Documentation]({docs_url})", unsafe_allow_html=True)
        else:
            st.caption("Documentation not built yet. Run `make -C docs html`.")

    pg.run()


if __name__ == "__main__":
    main()
