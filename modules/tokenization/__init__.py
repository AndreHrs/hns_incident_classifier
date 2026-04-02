import re

from .patterns import TOKEN_PATTERN, DATE_TOKEN, TIME_TOKEN, ORDINAL_TOKEN, NUM_TOKEN
from .normalize import _normalise_time_formats, _collapse_natural_dates

def _get_tokenizer() -> re.Pattern:
    """Return the compiled domain-aware tokenizer pattern.

    :returns: Compiled regular expression used to split text into tokens.
    :rtype: re.Pattern
    """
    return TOKEN_PATTERN


def normalize_token(token: str) -> str:
    """Replace a single token with a semantic placeholder if it matches a
    known numeric, date, time, or ordinal pattern.

    Placeholder mapping:

    - ``<date>``  — ``DD/MM/YYYY``-style dates.
    - ``<time>``  — clock times such as ``16:14hrs`` or ``7:30am``.
    - ``<ord>``   — ordinal numbers such as ``1st``, ``24th``.
    - ``<num>``   — plain integers or decimals.

    Tokens that do not match any pattern are returned unchanged.

    :param token: A single token produced by the tokenizer.
    :type token: str
    :returns: Placeholder string or the original token.
    :rtype: str
    """
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
    """Normalise a list of tokens produced by :func:`handle_tokenization`.

    When *keep_numbers* is ``False``, multi-token natural-language dates are
    first collapsed into a single ``<date>`` placeholder via
    :func:`~modules.tokenization.normalize._collapse_natural_dates`, then each
    remaining token is passed through :func:`normalize_token`.

    :param tokens: Raw token list to normalise.
    :type tokens: list[str]
    :param keep_numbers: When ``True`` the token list is returned as-is,
        preserving numeric tokens. Defaults to ``True``.
    :type keep_numbers: bool
    :returns: Normalised token list with numeric/temporal placeholders applied
        (unless *keep_numbers* is ``True``).
    :rtype: list[str]
    """
    if keep_numbers:
        return tokens
    tokens = _collapse_natural_dates(tokens) # collapse the dates first
    return [normalize_token(t) for t in tokens]


def handle_tokenization(
    text: str
) -> list[str]:
    """Tokenize text into domain-friendly tokens.

    A pre-pass via
    :func:`~modules.tokenization.normalize._normalise_time_formats` converts
    informal time variants (e.g. ``"6.30 am"``, ``"1614 hrs"``) to a
    canonical colon form before the main regex tokenizer runs.

    :param text: Pre-processed input string to tokenize.
    :type text: str
    :returns: Ordered list of tokens extracted from *text*.
    :rtype: list[str]
    """
    text = _normalise_time_formats(text)   # pre-pass for time variants
    regex = _get_tokenizer()
    return regex.findall(text)