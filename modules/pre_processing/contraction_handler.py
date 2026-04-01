
import contractions
import pandas as pd


def handle_contraction(text: str) -> str:
    """Expand contractions in a given text string.

    Args:
        text: Input text that may contain contractions.

    Returns:
        Text with contractions expanded.
    """
    if not isinstance(text, str):
        return text
    return contractions.fix(text)


def apply_contraction_handler(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Apply contraction handling to a DataFrame column.

    Args:
        df: Input DataFrame.
        column: Column name to apply contraction handling to.

    Returns:
        DataFrame with contractions expanded in the specified column.
    """
    df[column] = df[column].apply(handle_contraction)
    return df