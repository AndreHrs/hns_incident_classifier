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
    r"^(?:\d{1,2}[:.]\d{2}(?:hrs?|am|pm)?|\d{3,4}hrs?|\d{1,2}(?:am|pm))$",
    re.IGNORECASE,
)

ORDINAL_TOKEN = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)
NUM_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")


# ---------------------------------------------------------------------------
# Pre-tokenization: normalise time format variants
#
# Incident reports use several informal time styles that would otherwise
# tokenize incorrectly. We normalise them all to the colon form that
# TIME_TOKEN already handles (e.g. "16:14hrs") before tokenizing.
#
# Handles:
#   "6.30 am"       → "6:30am"   (dot-separated am/pm time)
#   "7.00 – 7.30 am"→ "7:00 - 7:30am"  (leading half of a range)
#   "16.14hrs"      → "16:14hrs" (dot-separated 24h time)
#   "16:14 hrs"     → "16:14hrs" (space before hrs)
#   "1614 hrs"      → "1614hrs"  (space before hrs in military time)
# ---------------------------------------------------------------------------
 
# Leading time in a range: "7.00 – 7.30 am" — the "7.00" part has no am/pm
# directly after it, so we need to fix it before the trailing time gets fixed.
_DOT_TIME_RANGE_LEAD_RE = re.compile(
    r'\b(\d{1,2})\.(\d{2})\s*-\s*(?=\d{1,2}[:.]\d{2})',
    re.IGNORECASE,
)
 
# "6.30 am", "7.30 pm", "6.30am" (zero-width space ok)
_DOT_TIME_RE = re.compile(
    r'\b(\d{1,2})\.(\d{2})\s*(am|pm)\b',
    re.IGNORECASE,
)
 
# "16.14hrs", "16.14hr", "16.14 hrs", "16.14 hr"
_DOT_HOUR_RE = re.compile(
    r'\b(\d{1,4})\.(\d{2})\s*(hrs?)\b',
    re.IGNORECASE,
)
 
# "16:14 hrs", "16:14 hr"
_SPACE_HRS_RE = re.compile(
    r'\b(\d{1,2}:\d{2})\s+(hrs?)\b',
    re.IGNORECASE,
)
 
# "1614 hrs", "1614 hr"
_BARE_HHMM_HRS_RE = re.compile(
    r'\b(\d{3,4})\s+(hrs?)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Post-tokenization: collapse natural-language dates
#
# "9th January 2018" tokenizes to ['9th', 'january', '2018'] — three
# separate tokens that individually look like <ord>, word, <num>.
# We scan for the pattern (ordinal|day-number) + month-name + 4-digit-year
# and collapse it to a single <date> token.
#
# Also handles month-first: "January 9th 2018".
#
# NOTE: must run on RAW tokens, before normalize_token(), because we need
# to read the ordinal suffix and month name. Running after normalize_token
# would give us ['<ord>', 'january', '<num>'] and the pattern is lost.
# ---------------------------------------------------------------------------
 
_MONTHS = {
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
}
_ORDINAL_RE = re.compile(r'^\d+(?:st|nd|rd|th)$', re.IGNORECASE)
_DAY_NUM_RE = re.compile(r'^([1-9]|[12]\d|3[01])$')
_YEAR_RE = re.compile(r'^(19|20)\d{2}$')