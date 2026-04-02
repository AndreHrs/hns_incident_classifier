import re


def basic_text_cleanup(text: str) -> str:
    """Normalize quotes, dashes, and whitespace without changing token meaning."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def handle_numeric(text: str, keep_numbers: bool = True) -> str:
    """Light text cleaning only.

    Numeric normalization is intentionally not performed here.
    It will be applied after tokenization at token level.

    Args:
        text: Input text.
        keep_numbers: Kept only for backward compatibility.

    Returns:
        Cleaned, lowercased string.
    """
    if not isinstance(text, str):
        text = str(text)

    text = basic_text_cleanup(text)
    text = text.lower()
    return text