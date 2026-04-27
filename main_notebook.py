# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: torch-rocm
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
## Need to install contractions. Need to list this!
# # %pip install contractions

# %% [markdown]
# ## Notes
# Please do not that this is residual code.
# I start experimenting and rapidly prototype on .ipynb then after it is fixed and complete only then I move it to external module.
#
# Exercise caution on using these code as references
#

# %%
from modules import OneTextPreProcessor

import json

# Open and read the file
with open('column_map.json', 'r') as file:
    column_map = json.load(file)

# If wanted to use spacy transformer model, set to en_core_web_trf (better result, at 11x slower tradeoff)
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

# %%
import torch
from experiment_setup.tf_idf_runner import tf_idf_run_multiple, tf_idf_hparam_search

train_df = pd.read_csv("dataset/model1_train.csv")
valid_df = pd.read_csv("dataset/model1_valid.csv")
test_df = pd.read_csv("dataset/model1_test.csv")

text_col = column_map["Detailed Description of Event"]         # "description"

_EPOCHS = 100

# Baseline: plain TF-IDF features (current default)
tfidf_train_config = {
    "epochs": _EPOCHS,
    "patience": 12,
    "best_metric": "f1_macro",
    # Optimizer factory — receives the model after it is built inside the runner
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
    # CosineAnnealingLR: lr decays from initial to eta_min over T_max epochs
    "scheduler_fn": lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=_EPOCHS, eta_min=1e-6
    ),
    "scheduler_step_per_batch": False,
}

models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True, n=5,
    train_config=tfidf_train_config,
)

# Optional variant: TF-IDF-weighted SafetyBERT static embeddings.
# Keeps the TF-IDF runner intact; only switches the feature representation.
tfidf_safetybert_train_config = {
    **tfidf_train_config,
    "feature_representation": "tfidf_embed_avg",
    "embedding_model_name": "adanish91/safetybert",
}

safetybert_models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True, n=5,
    train_config=tfidf_safetybert_train_config,
)

# %%
# models

# %% [markdown]
# ## TF-IDF Hyperparameter Search (Optuna)
# Searches over: `lr`, `hidden_dim`, `epochs`, `scheduler`
# Pre-processing runs once before the study; each trial trains a fresh model.

# %%
hparam_study = tf_idf_hparam_search(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True,
    n_trials=40,
    timeout=None,   # set to seconds e.g. 3600 to cap wall time
)

print("Best val f1_macro :", hparam_study.best_value)
print("Best params       :", hparam_study.best_params)

# %%
# Visualise results — requires plotly: pip install plotly
import optuna

optuna.visualization.plot_optimization_history(hparam_study).show()
optuna.visualization.plot_param_importances(hparam_study).show()
optuna.visualization.plot_parallel_coordinate(hparam_study).show()

# %%
# Retrain n times with the best found params
best = hparam_study.best_params
_BEST_EPOCHS = best["epochs"]

best_train_config = {
    "epochs": _BEST_EPOCHS,
    "hidden_dim": best["hidden_dim"],
    "patience": 15,
    "best_metric": "f1_macro",
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=best["lr"]),
    "scheduler_fn": lambda opt: (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_BEST_EPOCHS, eta_min=1e-6)
        if best["scheduler"] == "cosine"
        else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, _BEST_EPOCHS // 5), gamma=0.5)
        if best["scheduler"] == "step"
        else None
    ),
    "scheduler_step_per_batch": False,
}

best_models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=True, n=5,
    train_config=best_train_config,
)

# %% [markdown]
# # Search for TF-IDF Potential Damage Model

# %%
train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

# %%
valid_df["Type of Potential Damage"].value_counts()

# %%
train_df = pd.read_csv("dataset/model2_train.csv")
valid_df = pd.read_csv("dataset/model2_valid.csv")
test_df = pd.read_csv("dataset/model2_test.csv")

hparam_study = tf_idf_hparam_search(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=False,
    n_trials=40,
    timeout=None,   # set to seconds e.g. 3600 to cap wall time
)

print("Best val f1_macro :", hparam_study.best_value)
print("Best params       :", hparam_study.best_params)
# Visualise results — requires plotly: pip install plotly
import optuna

optuna.visualization.plot_optimization_history(hparam_study).show()
optuna.visualization.plot_param_importances(hparam_study).show()
optuna.visualization.plot_parallel_coordinate(hparam_study).show()
# Retrain n times with the best found params
best = hparam_study.best_params
_BEST_EPOCHS = best["epochs"]

best_train_config = {
    "epochs": _BEST_EPOCHS,
    "hidden_dim": best["hidden_dim"],
    "patience": 15,
    "best_metric": "f1_macro",
    "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=best["lr"]),
    "scheduler_fn": lambda opt: (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_BEST_EPOCHS, eta_min=1e-6)
        if best["scheduler"] == "cosine"
        else torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, _BEST_EPOCHS // 5), gamma=0.5)
        if best["scheduler"] == "step"
        else None
    ),
    "scheduler_step_per_batch": False,
}

best_models = tf_idf_run_multiple(
    train_df, valid_df, test_df, text_col,
    keep_numbers=False, lemma_config=lemma_config,
    energy_model=False, n=5,
    train_config=best_train_config,
)

# %%
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import numpy as np

# %%
# ── Load once, reuse for both extraction methods ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
safety_bert = AutoModel.from_pretrained("adanish91/safetybert")


# %%
# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: Static token embeddings  (for Bi-GRU embedding matrix init)
# Pull the raw embedding weight matrix — shape: (vocab_size=30522, hidden=768)
# This is a direct lookup table, no context.
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding_matrix(vocab: dict[str, int]) -> torch.Tensor:
    """
    Build an embedding matrix aligned to vocabulary (NEEDS {word: index} dictionary)

    Args:
        vocab: {word: index} dict

    Returns:
        matrix: Tensor of shape (len(vocab), 768)
    """
    static_embeddings = safety_bert.embeddings.word_embeddings.weight.detach()
    # shape: (30522, 768) — one vector per BERT subword token

    matrix = torch.zeros(len(vocab), 768)
    found, oov = 0, []

    for word, idx in vocab.items():
        # BERT uses WordPiece — a word may split into multiple subword tokens
        subword_ids = tokenizer.encode(word, add_special_tokens=False)
        if subword_ids:
            # Average subword embeddings to get one vector per word
            matrix[idx] = static_embeddings[subword_ids].mean(dim=0)
            found += 1
        else:
            oov.append(word)

    print(f"Coverage: {found}/{len(vocab)} words "
          f"({100*found/len(vocab):.1f}%)")
    if oov:
        print(f"OOV sample: {oov[:10]}")
    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: Contextual sentence embeddings  (better input signal for Bi-GRU)
# Run each incident report through safetyBERT and get per-token hidden states.
# Your Bi-GRU then processes these instead of a simple embedding lookup.
# ─────────────────────────────────────────────────────────────────────────────
def get_contextual_embeddings(
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu"
) -> list[torch.Tensor]:
    """
    Encode a list of raw incident descriptions into contextual token embeddings.

    Args:
        texts:      List of raw description strings (before your preprocessing)
        batch_size: How many documents to encode at once
        max_length: BERT max tokens — 128 is fine for incident reports
        device:     'cuda' or 'cpu'

    Returns:
        List of tensors, each shape (seq_len, 768) — one per document.
        seq_len varies per document (padding removed).
    """
    safety_bert.to(device)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = safety_bert(**inputs)

        # last_hidden_state: (batch, seq_len, 768)
        hidden = outputs.last_hidden_state

        # Strip padding — return only real tokens per document
        for j, length in enumerate(inputs["attention_mask"].sum(dim=1)):
            # slice off [CLS] and [SEP] too — your Bi-GRU doesn't need them
            all_embeddings.append(hidden[j, 1:length-1, :].cpu())

    return all_embeddings


# %%
vocab = {
    "worker": 0,
    "fell": 1,
    "on": 2,
    "ladder": 3
}

matrix = get_embedding_matrix(vocab)
print(matrix.shape)  # → torch.Size([4, 768])
print(matrix)
