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

df = pd.read_csv("dataset/final_dataset.csv")
print(df.columns)

# %%
## Need to install contractions. Need to list this!
# %pip install contractions

# %%
from modules import OneTextPreProcessor

import json

# Open and read the file
with open('column_map.json', 'r') as file:
    column_map = json.load(file)

# oneTextPreProcessor = OneTextPreProcessor(keep_numbers=True, column_map=column_map)
# mod_df = oneTextPreProcessor.pre_process_df(df, column_map["Detailed Description of Event"])
# mod_df

# %%
# oneTextPreProcessor = OneTextPreProcessor(keep_numbers=False, column_map=column_map)
# mod_df = oneTextPreProcessor.pre_process_df(df, column_map["Detailed Description of Event"])
# mod_df

# %%
import os

data_dir = "dataset"
df_list = {}

# scan and read all CSV files
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(data_dir, file)
        
        df = pd.read_csv(file_path)
        
        # rename columns
        df = df.rename(columns=column_map)
        
        df_list[file] = df

def check_class_dist(df, col_name): 
  display(df[col_name].value_counts().to_frame(name="count"))


for (key, value) in df_list.items():
    print(f"Class distribution for {key}")
    print("Energy Type Distributions:")
    check_class_dist(value, "energy_type")
    print("Potantial Damage Distributions:")
    check_class_dist(value, "potential_damage")
    print("="*32)

# If wanted to use spacy transformer model, set to en_core_web_trf (better result, at 11x slower tradeoff)
lemma_config = {
    "spacy_model": "en_core_web_sm",
    "filter_stop_words": True,
    "short_tokens_threshold": 0,
    "use_ner": True,
}

# oneTextPreProcessor = OneTextPreProcessor(keep_numbers=True, column_map=column_map, lemmatize=True, lemma_config=lemma_config)
# mod_df = oneTextPreProcessor.pre_process_df(df, column_map["Detailed Description of Event"])
# mod_df

# %%
def pre_process(data_path):
    oneTextPreProcessor = OneTextPreProcessor(keep_numbers=False, column_map=column_map)
    proc_df = oneTextPreProcessor.pre_process_df(
        pd.read_csv(data_path),
        column_map["Detailed Description of Event"]
    )
    return proc_df

model1_train = pre_process("dataset/model1_train.csv")
model1_valid = pre_process("dataset/model1_valid.csv")
model1_test = pre_process("dataset/model1_test.csv")

model2_train = pre_process("dataset/model2_train.csv")
model2_valid = pre_process("dataset/model2_valid.csv")
model2_test = pre_process("dataset/model2_test.csv")

# %%
tokens_col = f"description_tokens_lemma"

# %%
from implementations.tf_idf import TFIDFClassifier, TFIDFVectorizer, build_tfidf_dataloader
from modules.training_loop import _build_train_config, training

# %%
import torch
from modules.encoding import LabelEncoder

text_col = column_map["Detailed Description of Event"]         # "description"
label_col = "energy_type"  # or "Potential Damage" for model2

train_tokenized_docs = model1_train[tokens_col].tolist()
val_tokenized_docs   = model1_valid[tokens_col].tolist()

label_enc = LabelEncoder()
label_enc.fit(model1_train[label_col].tolist())

train_labels = torch.tensor(label_enc.encode_many(model1_train[label_col].tolist()))
val_labels   = torch.tensor(label_enc.encode_many(model1_valid[label_col].tolist()))

# --- TF-IDF ---
vectorizer = TFIDFVectorizer().fit(train_tokenized_docs)
train_vecs = vectorizer.transform(train_tokenized_docs)
val_vecs   = vectorizer.transform(val_tokenized_docs)

train_dl = build_tfidf_dataloader(train_vecs, train_labels)
val_dl   = build_tfidf_dataloader(val_vecs, val_labels, shuffle=False)

num_classes = label_enc.num_classes
model = TFIDFClassifier(vocab_size=len(vectorizer.vocab), num_classes=num_classes)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

config = _build_train_config(
    model_type="tf_idf",
    model=model,
    train_dl=train_dl,
    valid_dl=val_dl,
    epochs=100,
    device=device,
    patience=12,               # required — was missing
    criterion_weights=None,   # required — None means equal class weights
    best_metric="f1_macro",
    need_length=False,
    energy_model=True,        # True = use Energy labels, False = Risk labels
)
results = training(**config)

# %%
from sklearn.metrics import classification_report

model = results["config"]["model"]
valid_dl = results["config"]["valid_dl"]
device = results["config"]["device"]

model.eval()
all_preds, val_labels = [], []

with torch.no_grad():
    for batch in valid_dl:
        D, _, Energy, Risk = batch
        D = D.to(device)
        logits = model(D)
        all_preds.append(logits.argmax(dim=1).cpu())
        val_labels.append(Energy.cpu())  # or Risk if energy_model=False

preds = torch.cat(all_preds).numpy()
val_labels = torch.cat(val_labels).numpy()

labels = list(range(label_enc.num_classes))
target_names = label_enc.decode_many(labels)

print(classification_report(val_labels, preds, labels=labels, target_names=target_names))

# %%
from sklearn.metrics import f1_score
import numpy as np

# find labels that actually appear in val set
present = np.unique(val_labels)
print(f1_score(val_labels, preds, labels=present, average='macro'))

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(val_labels, preds, labels=labels)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(14,12))
disp.plot(ax=ax, xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# %%
model1_valid.columns

# %%
# Look at Specialized Shape examples in validation
spec_shape = model1_valid[model1_valid['energy_type'] == 'Specialized Shape']

# Look at Susceptible Part examples in validation
susc_part = model1_valid[model1_valid['energy_type'] == 'Susceptible Part']

# %%
compare = pd.concat([spec_shape, susc_part])[['reference', 'energy_type', 'description', 'description_tokens']]
compare.to_csv("debugging.csv")

# %%
import pandas as pd
import torch
import numpy as np

# Get predictions with references
model.eval()
all_preds, all_true, all_refs = [], [], []

with torch.no_grad():
    for i, batch in enumerate(valid_dl):
        D, _, Energy, Risk = batch
        D = D.to(device)
        logits = model(D)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(Energy.numpy())

# Decode labels
pred_labels = label_enc.decode_many(all_preds)
true_labels = label_enc.decode_many(all_true)

# Build results df — we need to align with model1_valid by position
# valid_dl was built without shuffling, so order is preserved
results_df = model1_valid.copy().reset_index(drop=True)
results_df['pred_label'] = pred_labels
results_df['true_label'] = true_labels
results_df['correct'] = results_df['pred_label'] == results_df['true_label']

# ── Priority review pairs ──────────────────────────────────────────────────
review_pairs = [
    ("Specialized Shape", "Susceptible Part"),
    ("Susceptible Part",  "Specialized Shape"),
    ("Machine",           "Vehicular"),
    ("Vehicular",         "Machine"),
    ("Gravitational",     "Human"),
    ("Human",             "Gravitational"),
    ("Object",            "Gravitational"),
    ("Gravitational",     "Object"),
    ("Thermal",           "Object"),
    ("Thermal",           "Other"),
]

cols = ['reference', 'true_label', 'pred_label', 'description', 'description_tokens']

for true_cls, pred_cls in review_pairs:
    mask = (
        (results_df['true_label'] == true_cls) &
        (results_df['pred_label'] == pred_cls)
    )
    subset = results_df[mask][cols]
    if subset.empty:
        continue



# %%
# Export all misclassified priority pairs to CSV for client review
flagged_rows = []
for true_cls, pred_cls in review_pairs:
    mask = (
        (results_df['true_label'] == true_cls) &
        (results_df['pred_label'] == pred_cls)
    )
    flagged_rows.append(results_df[mask][cols])

flagged_df = pd.concat(flagged_rows).drop_duplicates(subset='reference')
flagged_df.to_csv("labelling_review_flagged.csv", index=False)
print(f"Exported {len(flagged_df)} flagged cases to labelling_review_flagged.csv")
