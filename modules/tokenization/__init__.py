import re

from .patterns import TOKEN_PATTERN, DATE_TOKEN, TIME_TOKEN, ORDINAL_TOKEN, NUM_TOKEN
from .normalize import _normalise_time_formats, _collapse_natural_dates

def _get_tokenizer() -> re.Pattern:
    """Return compiled tokenizer."""
    return TOKEN_PATTERN


def normalize_token(token: str) -> str:
    """Normalize only tokens that are clearly numeric/date/time-like."""
    if DATE_TOKEN.match(token):
        return "<date>"
    if TIME_TOKEN.match(token):
        return "<time>"
    if ORDINAL_TOKEN.match(token):
        return "<ord>"
    if NUM_TOKEN.match(token):
        return "<num>"
    return token


def normalize_tokens(tokens: list[str], keep_numbers: bool = True) -> list[str]:
    """Normalize token list after tokenization."""
    if keep_numbers:
        return tokens
    tokens = _collapse_natural_dates(tokens) # collapse the dates first
    return [normalize_token(t) for t in tokens]


def handle_tokenization(
    text: str,
    keep_decimals_together: bool = True,
    split_num_units: bool = False,
) -> list[str]:
    """Tokenize text into domain-friendly tokens.

    The extra arguments are kept for backward compatibility.
    """
    text = _normalise_time_formats(text)   # pre-pass for time variants
    regex = _get_tokenizer()
    return regex.findall(text)