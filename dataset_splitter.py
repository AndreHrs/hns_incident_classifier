"""Dataset splitter module."""

import pandas as pd
from sklearn.model_selection import train_test_split


def save_and_split_stratified(
    df: pd.DataFrame,
    model_name: str,
    stratify_by: str,
    random_seed: int = 42,
) -> None:
    """Split a DataFrame into train/validation/test CSVs using stratified sampling.

    Classes with fewer than 7 samples cannot be stratified across a 70/30 split,
    so they are excluded from the split and appended whole to the train set.

    The output files are written to ``dataset/<model_name>_{train,valid,test}.csv``.

    Args:
        df: Input DataFrame to split.
        model_name: Prefix used when naming the output CSV files.
        stratify_by: Column name to use for stratification.
        random_seed: Random seed passed to both train_test_split calls to ensure reproducibility. Defaults to 42.
    """
    counts = df[stratify_by].value_counts()
    # Need round(n * 0.30) >= 2 for second split to work, so n >= 7
    rare_classes = counts[counts < 7].index

    rare = df[df[stratify_by].isin(rare_classes)]
    common = df[~df[stratify_by].isin(rare_classes)]

    train, temp = train_test_split(
        common, test_size=0.30, stratify=common[stratify_by], random_state=random_seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[stratify_by], random_state=random_seed
    )

    train = pd.concat([train, rare], ignore_index=True)

    train.to_csv(f"dataset/{model_name}_train.csv", index=False)
    val.to_csv(f"dataset/{model_name}_valid.csv", index=False)
    test.to_csv(f"dataset/{model_name}_test.csv", index=False)

    print(f"Rare classes appended to train: {list(rare_classes)}")
    print(f"Train: {len(train)}, Valid: {len(val)}, Test: {len(test)}")


df = pd.read_csv("dataset/final_dataset.csv")

save_and_split_stratified(df, "model1", stratify_by="Energy Type")
save_and_split_stratified(df, "model2", stratify_by="Type of Potential Damage")
