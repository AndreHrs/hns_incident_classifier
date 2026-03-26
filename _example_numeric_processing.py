# %%
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath("../.."))

from modules.pre_processing.numeric_handler import handle_numeric
from modules.tokenization import handle_tokenization

def process_datasets(datasets, keep_numbers=True, keep_decimals_together=True, split_num_units=False):
    for df in datasets:
      clean_col = "events_clean"
      token_col = "events_tokens"

      df[clean_col] = df["events"].apply(lambda s: handle_numeric(s, keep_numbers=keep_numbers))
      df[token_col] = df[clean_col].apply(
          lambda s: handle_tokenization(s, keep_decimals_together, split_num_units)
      )

    return datasets


sample_dataset = pd.DataFrame({
    "events": ["the object is 10cm long", "he run 5.3km in 20.5 minutes", "water boils at 100°c", "visit us at https://example.com", "price dropped by 100% unbelieveable"]
})


print("\n=== CASE 1: keep_numbers=True, keep_decimals_together=True, split_num_units=True ===")
process_datasets([sample_dataset], keep_numbers=True, keep_decimals_together=True, split_num_units=True)
print(sample_dataset)

print("\n=== CASE 2: keep_numbers=True, keep_decimals_together=True, split_num_units=False ===")
process_datasets([sample_dataset], keep_numbers=True, keep_decimals_together=True, split_num_units=False)
print(sample_dataset)

print("\n=== CASE 3: keep_numbers=False, keep_decimals_together=False, split_num_units=True ===")
process_datasets([sample_dataset], keep_numbers=False, keep_decimals_together=False, split_num_units=True)
print(sample_dataset)

print("\n=== CASE 4: keep_numbers=False, keep_decimals_together=False, split_num_units=False ===")
process_datasets([sample_dataset], keep_numbers=False, keep_decimals_together=False, split_num_units=False)
print(sample_dataset)

print("\n=== CASE 5: keep_numbers=True, keep_decimals_together=False, split_num_units=True ===")
process_datasets([sample_dataset], keep_numbers=True, keep_decimals_together=False, split_num_units=True)
print(sample_dataset)

# Makes no sense to split decimals then keep number. Turns 5.1km to [5, ., 1km]
print("\n=== CASE 6: keep_numbers=True, keep_decimals_together=False, split_num_units=False ===")
process_datasets([sample_dataset], keep_numbers=True, keep_decimals_together=False, split_num_units=False)
print(sample_dataset)



# %%
