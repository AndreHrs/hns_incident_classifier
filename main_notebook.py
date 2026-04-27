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

# _EPOCHS = 100
# tfidf_train_config = {
#     "epochs": _EPOCHS,
#     "patience": 12,
#     "best_metric": "f1_macro",
#     # Optimizer factory — receives the model after it is built inside the runner
#     "optimizer_fn": lambda model: torch.optim.Adam(model.parameters(), lr=1e-3),
#     # CosineAnnealingLR: lr decays from initial to eta_min over T_max epochs
#     "scheduler_fn": lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
#         opt, T_max=_EPOCHS, eta_min=1e-6
#     ),
#     "scheduler_step_per_batch": False,
# }

# models = tf_idf_run_multiple(
#     train_df, valid_df, test_df, text_col,
#     keep_numbers=False, lemma_config=lemma_config,
#     energy_model=True, n=5,
#     train_config=tfidf_train_config,
# )

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
values = [t.value for t in hparam_study.trials if t.value is not None]
print(f"Unique objective values: {len(set(values))} / {len(values)}")


# %%
# import torch
# from modules.encoding import LabelEncoder

# text_col = column_map["Detailed Description of Event"]         # "description"
# label_col = "energy_type"  # or "Potential Damage" for model2

# train_tokenized_docs = model1_train[tokens_col].tolist()
# val_tokenized_docs   = model1_valid[tokens_col].tolist()
# test_tokenized_docs  = model1_test[tokens_col].tolist()

# label_enc = LabelEncoder()
# label_enc.fit(model1_train[label_col].tolist())

# train_labels = torch.tensor(label_enc.encode_many(model1_train[label_col].tolist()))
# val_labels   = torch.tensor(label_enc.encode_many(model1_valid[label_col].tolist()))
# test_labels  = torch.tensor(label_enc.encode_many(model1_test[label_col].tolist()))

# # --- TF-IDF ---
# vectorizer = TFIDFVectorizer().fit(train_tokenized_docs)
# train_vecs = vectorizer.transform(train_tokenized_docs)
# val_vecs   = vectorizer.transform(val_tokenized_docs)
# test_vecs  = vectorizer.transform(test_tokenized_docs)

# train_dl = build_tfidf_dataloader(train_vecs, train_labels)
# val_dl   = build_tfidf_dataloader(val_vecs, val_labels, shuffle=False)
# test_dl  = build_tfidf_dataloader(test_vecs, test_labels, shuffle=False)

# num_classes = label_enc.num_classes
# model = TFIDFClassifier(vocab_size=len(vectorizer.vocab), num_classes=num_classes, hidden_dim=256)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# results = training(
#     model_type="tf_idf",
#     model=model,
#     train_dl=train_dl,
#     valid_dl=val_dl,
#     test_dl=test_dl,
#     epochs=100,
#     device=device,
#     patience=12,
#     criterion_weights=None,
#     best_metric="f1_macro",
#     need_length=False,
#     energy_model=True,
#     num_classes=num_classes,
# )

# %%

# # import pandas as pd

# # true_class = "Susceptible Part"
# # pred_class = "Specialized Shape"

# # true_idx = label_enc.encode(true_class)
# # pred_idx = label_enc.encode(pred_class)

# # mask = (val_labels == true_idx) & (preds == pred_idx)
# # misclassified_indices = mask.nonzero()[0]

# # text_col = column_map["Detailed Description of Event"]
# # misclassified_texts = model1_valid.iloc[misclassified_indices][[text_col, "energy_type"]].copy()
# # misclassified_texts["predicted"] = pred_class

# # print(f"Found {len(misclassified_indices)} sample(s) labelled '{true_class}' but predicted as '{pred_class}':\n")
# # pd.set_option("display.max_colwidth", None)
# # misclassified_texts.reset_index(drop=True)

# # # Look at Specialized Shape examples in validation
# # spec_shape = model1_valid[model1_valid['energy_type'] == 'Specialized Shape']

# # # Look at Susceptible Part examples in validation
# # susc_part = model1_valid[model1_valid['energy_type'] == 'Susceptible Part']

# compare = pd.concat([spec_shape, susc_part])[['reference', 'energy_type', 'description', 'description_tokens']]
# compare.to_csv("debugging.csv")

# import pandas as pd
# from modules.inference import run_inference

# # Run inference on the validation set
# infer_config = {
#     "model": model,
#     "device": device,
#     "need_length": False,
#     "energy_model": True,
#     "test_dl": val_dl,
# }
# infer_result = run_inference(infer_config)

# all_preds = infer_result["all_preds"].numpy()
# all_true  = infer_result["all_targets"].numpy()

# # Decode labels
# pred_labels = label_enc.decode_many(all_preds)
# true_labels = label_enc.decode_many(all_true)

# # Build results df — valid_dl was built without shuffling, so order is preserved
# results_df = model1_valid.copy().reset_index(drop=True)
# results_df['pred_label'] = pred_labels
# results_df['true_label'] = true_labels
# results_df['correct'] = results_df['pred_label'] == results_df['true_label']

# # ── Priority review pairs ──────────────────────────────────────────────────
# review_pairs = [
#     ("Specialized Shape", "Susceptible Part"),
#     ("Susceptible Part",  "Specialized Shape"),
#     ("Machine",           "Vehicular"),
#     ("Vehicular",         "Machine"),
#     ("Gravitational",     "Human"),
#     ("Human",             "Gravitational"),
#     ("Object",            "Gravitational"),
#     ("Gravitational",     "Object"),
#     ("Thermal",           "Object"),
#     ("Thermal",           "Other"),
# ]

# cols = ['reference', 'true_label', 'pred_label', 'description', 'description_tokens']

# for true_cls, pred_cls in review_pairs:
#     mask = (
#         (results_df['true_label'] == true_cls) &
#         (results_df['pred_label'] == pred_cls)
#     )
#     subset = results_df[mask][cols]
#     if subset.empty:
#         continue

# valid_df = model1_valid.copy().reset_index(drop=True)
# valid_df["pred_label"] = pred_labels
# valid_df["true_label"] = true_labels
# valid_df["correct"]    = valid_df["pred_label"] == valid_df["true_label"]

# print(f"Test accuracy : {valid_df['correct'].mean():.3f}  ({valid_df['correct'].sum()}/{len(valid_df)})")
# valid_df.head(3)

# # ── Inference on test set ─────────────────────────────────────────────────
# test_tokenized_docs = model1_test[tokens_col].tolist()
# test_labels_enc     = torch.tensor(label_enc.encode_many(model1_test[label_col].tolist()))
# test_vecs           = vectorizer.transform(test_tokenized_docs)
# test_dl             = build_tfidf_dataloader(test_vecs, test_labels_enc, shuffle=False)

# test_infer_config = {
#     "model": model,
#     "device": device,
#     "need_length": False,
#     "energy_model": True,
#     "test_dl": test_dl,
# }
# test_result = run_inference(test_infer_config)

# test_pred_labels = label_enc.decode_many(test_result["all_preds"].numpy())
# test_true_labels = label_enc.decode_many(test_result["all_targets"].numpy())

# test_df = model1_test.copy().reset_index(drop=True)
# test_df["pred_label"] = test_pred_labels
# test_df["true_label"] = test_true_labels
# test_df["correct"]    = test_df["pred_label"] == test_df["true_label"]

# print(f"Test accuracy : {test_df['correct'].mean():.3f}  ({test_df['correct'].sum()}/{len(test_df)})")
# test_df.head(3)



# %%
# # Export all misclassified priority pairs to CSV for client review
# flagged_rows = []
# for true_cls, pred_cls in review_pairs:
#     mask = (
#         (results_df['true_label'] == true_cls) &
#         (results_df['pred_label'] == pred_cls)
#     )
#     flagged_rows.append(results_df[mask][cols])

# flagged_df = pd.concat(flagged_rows).drop_duplicates(subset='reference')
# flagged_df.to_csv("labelling_review_flagged.csv", index=False)
# print(f"Exported {len(flagged_df)} flagged cases to labelling_review_flagged.csv")
