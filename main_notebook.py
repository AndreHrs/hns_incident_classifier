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

oneTextPreProcessor = OneTextPreProcessor(keep_numbers=True, column_map=column_map)
mod_df = oneTextPreProcessor.pre_process_df(df, column_map["Detailed Description of Event"])
mod_df

# %%
oneTextPreProcessor = OneTextPreProcessor(keep_numbers=False, column_map=column_map)
mod_df = oneTextPreProcessor.pre_process_df(df, column_map["Detailed Description of Event"])
mod_df
