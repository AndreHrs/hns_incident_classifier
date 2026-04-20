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
#     display_name: nlp
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

