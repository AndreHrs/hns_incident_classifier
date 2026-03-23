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

# %% [markdown]
# # Pre Processing
#
# If you have problem with cannot import modules. Try restarting the jupyter kernel before re-running. Usually it is because of the cache.

# %% [markdown]
# ## Cleaning and normalization

# %%
from modules.pre_processing import pre_process_dataset

# Just call pre process
some_data = pre_process_dataset()

# %% [markdown]
# ## Tokenization

# %%
from modules.tokenization import handle_tokenization

some_data = handle_tokenization()

# %% [markdown]
# ## Noise and Outlier Cleaning

# %%
from modules.outlier_removal import remove_outliers

some_data = remove_outliers()

# %% [markdown]
# ## Encoding and Vectorization

# %%
from modules.encoding import handle_encoding

some_data = handle_encoding()

# %% [markdown]
# ## Word Embedding

# %%
from modules.embedding import handle_embedding

some_data = handle_embedding()
