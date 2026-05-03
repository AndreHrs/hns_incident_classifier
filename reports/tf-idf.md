# TF‑IDF Results (Leaderboard Summary)

This report summarizes TF‑IDF performance using runs recorded in `leaderboard/leaderboard.csv`, and highlights the top-performing runs for each TF‑IDF task variant.

## Overall performance (all TF‑IDF runs)

- **Number of runs**: 90
- **Validation macro‑F1 (primary metric)**:
  - **Mean**: 0.5825 (std: 0.0673)
  - **Min / Median / Max**: 0.2290 / 0.6015 / 0.6420
  - **25% / 75% quantiles**: 0.5763 / 0.6136
- **Validation accuracy**:
  - **Mean**: 0.6983
  - **Max**: 0.7875
- **Validation loss**:
  - **Best (min)**: 0.5098
- **Training time**:
  - **Mean**: 1.075 s
  - **Max**: 4.596 s

### Benchmark takeaway

Across all TF‑IDF runs, the **best validation macro‑F1 is ~63%** (max: 0.6420; upper quartile: 0.6136). We treat **~63% macro‑F1** as a **baseline benchmark for “basic functionality”** of a TF‑IDF model on this project.

## Top performance (per task)

### Best for potential damage (`energy_model=False`)

- **Run name**: `tf_idf_run_20260427_174823`
- **Val macro‑F1**: **0.6420**
- **Val accuracy**: 0.7875
- **Val weighted‑F1**: 0.7811
- **Val loss**: 0.5098
- **Training details**: `epochs_max=60`, `patience=15`, `best_epoch=2`, `lr=0.009772`, scheduler=`StepLR`
- **Saved model**: `trained_models/20260427_174823_tf_idf/tf_idf_model.pt`

### Best for energy type (`energy_model=True`)

- **Run name**: `tf_idf_run_20260427_174505`
- **Val macro‑F1**: **0.6371**
- **Val accuracy**: 0.6667
- **Val weighted‑F1**: 0.6596
- **Val loss**: 1.1210
- **Training details**: `epochs_max=80`, `patience=15`, `best_epoch=6`, `lr=0.001540`, scheduler=`StepLR`
- **Saved model**: `trained_models/20260427_174505_tf_idf/tf_idf_model.pt`

## Best parameters (from notebook output)

The notebook code (`main_notebook.py`) performs an Optuna search and prints:
`print("Best params       :", hparam_study.best_params)`.

However, in the current saved `main_notebook.ipynb` content, the printed Optuna *output values* (the actual dict of best params) are not present/searchable, so the exact Optuna `best_params` dict cannot be recovered from the notebook artifact alone.

What we *can* report from the **best runs above** (these are logged per-run in `leaderboard/leaderboard.csv`):

- **lr**: 0.0097721189210879
- **epochs_max**: 60
- **patience**: 15
- **scheduler**: `StepLR` (stepped per epoch; `scheduler_step_per_batch=False`)
- **energy_model**: False

Note: Optuna also searches over `hidden_dim` (and other items), but `hidden_dim` is not currently recorded in `leaderboard/leaderboard.csv`. If needed, we should either (a) re-run the Optuna cell to re-print `best_params`, or (b) log `hidden_dim` into the leaderboard for each TF‑IDF run.

## Next Step
Compare whether embedding from SafetyBERT improves the performance or not