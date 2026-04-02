from .patterns import _DOT_TIME_RANGE_LEAD_RE, _DOT_TIME_RE ,_DOT_HOUR_RE ,_SPACE_HRS_RE ,_BARE_HHMM_HRS_RE ,_ORDINAL_RE, _DAY_NUM_RE, _MONTHS, _YEAR_RE

def _normalise_time_formats(text: str) -> str:
    """Normalise informal time variants before tokenization.
 
    Order matters: range-lead must run before the trailing am/pm fix,
    and dot-hour before the space-hrs fix.
    """
    text = _DOT_TIME_RANGE_LEAD_RE.sub(r'\1:\2 - ', text)
    text = _DOT_TIME_RE.sub(r'\1:\2\3', text)
    text = _DOT_HOUR_RE.sub(r'\1:\2\3', text)
    text = _SPACE_HRS_RE.sub(r'\1\2', text)
    text = _BARE_HHMM_HRS_RE.sub(r'\1\2', text)
    return text

def _collapse_natural_dates(tokens: list[str]) -> list[str]:
    """Collapse 3-token natural-language dates into a single '<date>' token.
 
    Matches (on raw tokens, case-insensitive):
      - "<day_ord_or_num> <month> <4digit_year>"  e.g. "9th January 2018"
      - "<month> <day_ord_or_num> <4digit_year>"  e.g. "January 9th 2018"
    """
    result = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            t0, t1, t2 = tokens[i], tokens[i + 1], tokens[i + 2]
            # day-first: "9th January 2018" or "9 January 2018"
            if (_ORDINAL_RE.match(t0) or _DAY_NUM_RE.match(t0)) \
                    and t1.lower() in _MONTHS \
                    and _YEAR_RE.match(t2):
                result.append('<date>')
                i += 3
                continue
            # month-first: "January 9th 2018" or "January 9 2018"
            if t0.lower() in _MONTHS \
                    and (_ORDINAL_RE.match(t1) or _DAY_NUM_RE.match(t1)) \
                    and _YEAR_RE.match(t2):
                result.append('<date>')
                i += 3
                continue
        result.append(tokens[i])
        i += 1
    return result