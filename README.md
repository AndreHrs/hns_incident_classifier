# Capstone Project — Incident Classifier

Requires **conda** and **Python 3.12 or higher**.

---

## Table of Contents

- [User Guide](#user-guide)
  - [Setup](#setup)
  - [Running the Web UI](#running-the-web-ui)
  - [CLI Usage](#cli-usage)
- [Developer / Maintainer Guide](#developer--maintainer-guide)
  - [Code Style](#code-style)
  - [Naming Conventions](#naming-conventions)
  - [Development Workflow](#development-workflow)
  - [Notebook Synchronization](#notebook-synchronization)
  - [Module Placement](#module-placement)

---

# User Guide

## Setup

### Prerequisites

- **conda** — [Installation guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
- **Python 3.12+**

### Automatic (Recommended)

> **Note:** The installer scripts are tested on Linux. They should also work on macOS, but this has not been verified.

Run the installer once:

```bash
./install.sh
```

This will automatically detect your GPU, create a conda environment named `hs_classifier`, install all dependencies with the correct PyTorch build, and download the spaCy model.

To reinstall from scratch:

```bash
./install.sh --force
```

### Manual

Create and activate a conda environment, then install dependencies:

```bash
conda create -n hs_classifier python=3.12 -y
conda activate hs_classifier
pip install -r requirements.txt
```

Install the spaCy lemmatizer model:

```bash
python -m spacy download en_core_web_sm
```

**GPU acceleration (optional)**

For NVIDIA GPUs (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

For AMD GPUs (ROCm):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0
```

---

## Running the Web UI

**Automatic (Recommended)** will run the installer first if not yet set up:

```bash
./run_ui.sh
```

**Manual:**

```bash
streamlit run app/app.py
```

---

## CLI Usage

```
python cli.py [COMMAND] [OPTIONS]
```

Global help: `python cli.py -h`

### `train` — Fine-tune a model

```bash
python cli.py train \
  --train data/train.csv \
  --valid data/valid.csv \
  --test  data/test.csv \
  --model-type  <energy|damage> \
  --architecture <tf_idf|bigru|bert|looped_transformer>
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--text-col TEXT` | Column name for input text (default: `description`) |
| `--epochs N` | Number of training epochs |
| `--lr FLOAT` | Learning rate |
| `--patience N` | Early-stopping patience |
| `--hidden-dim N` | Hidden layer size |
| `--batch-size N` | Mini-batch size |
| `--embedding-type` | BiGRU only: `none`, `static`, or `contextual` |
| `--num-loops N` | Looped transformer only: number of loops |
| `--fine-tune` | Enable fine-tuning mode (flag, no value needed) |

Prints the artifacts directory and best tracked metric on completion.

### `infer` — Batch inference

```bash
python cli.py infer \
  --dataset data/unlabelled.csv \
  --output  results.csv \
  [--energy-model path/to/energy_run] \
  [--damage-model path/to/damage_run]
```

At least one of `--energy-model` or `--damage-model` must be supplied. The output CSV includes confidence tiers and action columns. Prints row counts and tier distributions.

### `metrics` — Inspect the leaderboard

```bash
# Show top 20 runs sorted by val_f1_macro
python cli.py metrics

# Filter and sort
python cli.py metrics --model-type energy --architecture bert --sort-by test_f1_macro --top 10

# Dump a single run's full details
python cli.py metrics --model-dir path/to/run
```

**Optional flags:**

| Flag | Description |
|---|---|
| `--sort-by COL` | Column to sort by (default: `val_f1_macro`) |
| `--ascending` | Sort ascending instead of descending |
| `--model-type` | Filter by `energy` or `damage` |
| `--architecture` | Filter by architecture name |
| `--top N` | Number of rows to show (default: `20`) |
| `--model-dir PATH` | Dump JSON details for a single saved run |

Renders a rich table if the `rich` package is installed, otherwise falls back to plain text.

---

## Building Documentation

```bash
cd docs
make html
```

---

# Developer / Maintainer Guide

## Code Style

Install the linting and hook tools:

```bash
pip install ruff pre-commit
```

Install the pre-commit hooks (run once after cloning):

```bash
pre-commit install
```

Ruff will now run automatically on every `git commit`. To run it manually:

```bash
# check for lint errors
python -m ruff check .

# auto-fix lint errors
python -m ruff check . --fix
```

> **Easy to miss:** After fixing lint errors, you must re-stage the changed files with `git add` before committing. Pre-commit only checks staged files — if the fixed files aren't staged, the commit will keep failing even though the errors are gone.

---

## Naming Conventions

| Type | Convention | Example |
|---|---|---|
| Variables | snake_case | `data_frame` |
| Functions | snake_case | `load_dataset()` |
| Constants | PascalCase | `ModelConfig` |
| Public Classes / Interfaces | PascalCase | `DataProcessor` |
| Private/Internal Functions | `_underscored_snake_case` | `_internal_helper()` |

Unused variables can use a bare underscore: `_, b = (0, 2.36)`

---

## Development Workflow

This repository follows a Git Flow–inspired workflow.

**Branch flow:**
```
dev → feature → PR → dev → release PR → main
```

### Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code. Protected — no direct pushes. |
| `dev` | Integration branch. All feature work merges here first. Protected. |
| `feature/...` | Short-lived branches created from `dev`. |

Feature branch naming format:
```
feature/#<ticketNo>-<short-description>
```
Example: `feature/#42-risk-classification`

### Starting a Feature

```bash
git checkout dev
git pull origin dev
git checkout -b feature/#<ticketNo>-<short-description>
```

### Before Creating a Pull Request

Sync your branch with the latest `dev` to avoid merge conflicts:

```bash
git checkout dev
git pull origin dev
git checkout feature/#<ticketNo>-<feature-name>
git merge dev
```

Open a Pull Request: `feature/... → dev`

Include in the PR description: what was implemented and any notes for reviewers. At least one team member approval is required.

### Sprint / Iteration Release

At the end of a sprint:

1. Create a PR: `dev → main`
2. Review and approve.
3. Merge into `main`.
4. Back-merge `main → dev` to keep them in sync.

---

## Notebook Synchronization

Install jupytext:

```bash
conda install jupytext -c conda-forge
```

The `main_notebook` is already configured as a paired notebook. To sync before and after making changes:

```bash
# Sync notebook from the paired .py file (before starting work)
jupytext --sync main_notebook.py

# Sync notebook after making changes
jupytext --sync main_notebook.ipynb
```

Always synchronize the notebook before committing. Failure to do so may cause merge conflicts or lost notebook changes.

---

## Module Placement

Keep the repository modular to minimize conflicts and keep the notebook concise.

```
├── main_notebook.ipynb
├── main_notebook.py
├── modules
│   ├── misc
│   │   └── example.py
│   └── pre_processing
├── README.md
```

Place reusable logic in modules rather than inline in the notebook:

```python
# modules/misc/example.py
def print_hello_world():
    print("Hello World")
```

```python
# In the notebook
from modules.misc.example import print_hello_world

print_hello_world()
```
