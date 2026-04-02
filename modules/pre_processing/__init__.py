import unicodedata
import re

from .contraction_handler import apply_contraction_handler
from modules.tokenization import handle_tokenization, normalize_tokens
import pandas as pd


class OneTextPreProcessor:
    """Pre-processor for text columns in a pandas DataFrame.

    Applies a standard NLP pre-processing pipeline: unicode normalisation,
    contraction expansion, tokenisation, and token normalisation.

    :param keep_numbers: When ``True``, numeric tokens are retained during
        normalisation. Defaults to ``False``.
    :type keep_numbers: bool
    :param column_map: Optional mapping of ``{original_name: new_name}`` used
        to rename DataFrame columns before processing. Defaults to ``{}``.
    :type column_map: dict[str, str]
    """

    def __init__(self, keep_numbers: bool = False, column_map: dict[str, str] = {}):
        self.keep_numbers = keep_numbers
        self.column_maps = column_map
    
    def _basic_text_cleanup(self, text: str) -> str:
        """Perform basic unicode and whitespace normalisation on a text string.

        Steps applied in order:

        1. NFKC unicode normalisation (fractions, ligatures, fullwidth chars).
        2. Curly quotes and em/en-dashes replaced with ASCII equivalents.
        3. Consecutive whitespace collapsed to a single space and stripped.
        4. Text lowercased.

        :param text: Raw input string to clean.
        :type text: str
        :returns: Cleaned, lowercased string.
        :rtype: str
        """
        # 1. NFKC first, normalize unicode fractions, ligatures, fullwidth
        text = unicodedata.normalize("NFKC", text)
        
        # 2. Normalize quotes, dashes, and whitespace without changing token meaning.
        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("–", "-").replace("—", "-")
        
        # 3. Whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        text = text.lower()

        return text
    
    def _rename_columns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Rename DataFrame columns according to ``self.column_maps``.

        Returns an unmodified copy when no column map was provided.

        :param data_frame: Input DataFrame whose columns may be renamed.
        :type data_frame: pandas.DataFrame
        :returns: Copy of *data_frame* with columns renamed as specified.
        :rtype: pandas.DataFrame
        """
        df = data_frame.copy()
        if self.column_maps:
            df.rename(columns=self.column_maps, inplace=True)
        return df

    def pre_process_df(self, data_frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Run the full pre-processing pipeline on a DataFrame text column.

        The following columns are added to the returned DataFrame:

        - ``<text_col>_clean`` — lowercased, unicode-normalised text.
        - ``<text_col>_tokens_raw`` — tokenised list before normalisation.
        - ``<text_col>`` (in-place) — contractions expanded.
        - ``<text_col>_tokens`` — final normalised token list.

        :param data_frame: Input DataFrame containing the text column.
        :type data_frame: pandas.DataFrame
        :param text_col: Name of the column that holds the raw text to process.
        :type text_col: str
        :returns: Copy of *data_frame* with the additional processed columns.
        :rtype: pandas.DataFrame
        """
        df = data_frame.copy()
        df = self._rename_columns(df)

        apply_contraction_handler(df, text_col)

        df[f"{text_col}_clean"] = df[text_col].astype(str).apply(
            lambda s: self._basic_text_cleanup(s)
        )

        df[f"{text_col}_tokens_raw"] = df[f"{text_col}_clean"].apply(
            lambda s: handle_tokenization(s)
        )

        df[f"{text_col}_tokens"] = df[f"{text_col}_tokens_raw"].apply(
            lambda toks: normalize_tokens(toks, keep_numbers=self.keep_numbers)
        )

        return df