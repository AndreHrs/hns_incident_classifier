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
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
import pandas as pd

df = pd.read_csv("dataset/final_dataset.csv")
print(df.columns)

# %%
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

from modules.pre_processing.numeric_handler import handle_numeric
from modules.tokenization import handle_tokenization, normalize_tokens

df = pd.read_csv("dataset/final_dataset.csv")

text_col = "Detailed Description of Event"

def test_preprocessing(df, keep_numbers, n=5):
    temp = df.copy()

    temp["description_clean"] = temp[text_col].astype(str).apply(
        lambda s: handle_numeric(s, keep_numbers=keep_numbers)
    )

    temp["description_tokens_raw"] = temp["description_clean"].apply(
        lambda s: handle_tokenization(s)
    )

    temp["description_tokens"] = temp["description_tokens_raw"].apply(
        lambda toks: normalize_tokens(toks, keep_numbers=keep_numbers)
    )

    print("\n" + "=" * 80)
    print(f"keep_numbers={keep_numbers}")
    print("=" * 80)

    for i, row in temp[[text_col, "description_clean", "description_tokens_raw", "description_tokens"]].head(n).iterrows():
        print(f"\nROW {i}")
        print("ORIGINAL:")
        print(row[text_col])
        print("\nCLEANED:")
        print(row["description_clean"])
        print("\nRAW TOKENS:")
        print(row["description_tokens_raw"])
        print("\nNORMALIZED TOKENS:")
        print(row["description_tokens"])
        print("-" * 80)

test_preprocessing(df, keep_numbers=True)
test_preprocessing(df, keep_numbers=False)

# %%
