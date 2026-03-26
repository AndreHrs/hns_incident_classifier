import re


def normalize_numbers_inline(s: str) -> str:
    """Replace numeric values with the ``<num>`` placeholder token.

    Handles two cases in order:

    1. Numbers glued to a unit (``10cm``, ``5%``, ``37°``) → ``<num>cm`` / ``<num>%``
    2. Standalone numbers → ``<num>``

    Args:
        s: Input string.

    Returns:
        String with numeric values replaced by ``<num>``.
    """
    s = re.sub(r"\b(\d+(?:\.\d+)?)(?=[a-zA-Z%°])", "<num>", s)
    s = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", s)
    return s


def handle_numeric(text: str, keep_numbers: bool = True) -> str:
    """Normalize a string by optionally replacing numbers and lowercasing.

    This is a string-in, string-out function suitable for use in a pipeline
    via ``Series.apply()``. Tokenization is intentionally excluded and handled
    separately by the tokenization module.

    Args:
        text: Input string to process.
        keep_numbers: If False, replace numeric values with the ``<num>`` token.

    Returns:
        Cleaned string, lowercased with numbers optionally replaced.
    """
    if not isinstance(text, str):
        text = str(text)
    if not keep_numbers:
        text = normalize_numbers_inline(text)
    return text.lower()
