import re


TOKEN_PATTERN = re.compile(
    r"""
    <num>|<date>|<time>|<ord>                  # placeholders, if already present
    |[a-z0-9]+(?:[/-][a-z0-9]+)+              # 220-101b, pil/jan18/003, pre-loaded
    |[a-z0-9]+(?:&[a-z0-9]+)+                 # d&b
    |[a-z]+\d+[a-z0-9]*                       # dt28, cr10, sop4209
    |\d+\.\d+[a-z]*                           # 3.5, 15.3km
    |\d+:\d+(?:hrs?|am|pm)?                   # 16:14hrs, 7:30am
    |\d+(?:st|nd|rd|th)                       # 24th, 9th
    |\d+[a-z]+                                # 1730hrs, 12v, 10cm
    |[a-z]+                                   # normal words
    |\d+                                      # integers
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)

DATE_TOKEN = re.compile(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
TIME_TOKEN = re.compile(
    r"^(?:\d{1,2}:\d{2}(?:hrs?|am|pm)?|\d{3,4}hrs|\d{1,2}(?:am|pm))$",
    re.IGNORECASE,
)
ORDINAL_TOKEN = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
NUM_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")


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
    return [normalize_token(tok) for tok in tokens]


def handle_tokenization(
    text: str,
    keep_decimals_together: bool = True,
    split_num_units: bool = False,
) -> list[str]:
    """Tokenize text into domain-friendly tokens.

    The extra arguments are kept for backward compatibility.
    """
    regex = _get_tokenizer()
    return regex.findall(text)