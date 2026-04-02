import unicodedata
import re

from .contraction_handler import apply_contraction_handler
from modules.tokenization import handle_tokenization, normalize_tokens
import pandas as pd


class OneTextPreProcessor:
    def __init__(self, keep_numbers: bool = False, column_map: dict[str, str] = {}):
        self.keep_numbers = keep_numbers
        self.column_maps = column_map
    
    def _basic_text_cleanup(self, text: str) -> str:
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
        df = data_frame.copy()
        if self.column_maps:
            df.rename(columns=self.column_maps, inplace=True)
        return df

    def pre_process_df(self, data_frame: pd.DataFrame, text_col: str) -> pd.DataFrame:
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