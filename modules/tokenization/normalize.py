"""Token normalization and natural-language date collapsing utilities."""

from .patterns import (
    _DOT_TIME_RANGE_LEAD_RE,
    _DOT_TIME_RE,
    _DOT_HOUR_RE,
    _SPACE_HRS_RE,
    _BARE_HHMM_HRS_RE,
    _ORDINAL_RE,
    _DAY_NUM_RE,
    _MONTHS,
    _YEAR_RE,
)


def _normalise_time_formats(text: str) -> str:
    """Normalise informal time variants to canonical colon form before tokenization.

    Substitutions applied (in order):

    1. Leading time in a range — ``"7.00 – 7.30 am"`` → ``"7:00 - 7:30am"``.
    2. Dot-separated am/pm time — ``"6.30 am"`` → ``"6:30am"``.
    3. Dot-separated 24 h time — ``"16.14hrs"`` → ``"16:14hrs"``.
    4. Space before *hrs* suffix — ``"16:14 hrs"`` → ``"16:14hrs"``.
    5. Military time with space — ``"1614 hrs"`` → ``"1614hrs"``.

    Order is significant: the range-lead fix must precede the trailing am/pm
    fix, and dot-hour must precede the space-hrs fix.

    :param text: Raw text string prior to tokenization.
    :type text: str
    :returns: Text with time variants rewritten to the canonical colon form.
    :rtype: str
    """
    text = _DOT_TIME_RANGE_LEAD_RE.sub(r"\1:\2 - ", text)
    text = _DOT_TIME_RE.sub(r"\1:\2\3", text)
    text = _DOT_HOUR_RE.sub(r"\1:\2\3", text)
    text = _SPACE_HRS_RE.sub(r"\1\2", text)
    text = _BARE_HHMM_HRS_RE.sub(r"\1\2", text)
    return text


def _collapse_natural_dates(tokens: list[str]) -> list[str]:
    """Collapse three-token natural-language dates into a single ``<date>`` placeholder.

    Scans *tokens* for consecutive triplets that match either of:

    - *day-first*  — ``"<ordinal-or-day> <month-name> <4-digit-year>"``,
      e.g. ``["9th", "January", "2018"]``.
    - *month-first* — ``"<month-name> <ordinal-or-day> <4-digit-year>"``,
      e.g. ``["January", "9th", "2018"]``.

    Matching is case-insensitive for month names and ordinal suffixes.

    .. note::
        Must be called on **raw** tokens, before :func:`normalize_token` runs,
        because it needs to inspect ordinal suffixes and month names.  After
        normalisation those become ``<ord>`` / ``<num>`` and the pattern is
        undetectable.

    :param tokens: Raw token list from :func:`~modules.tokenization.handle_tokenization`.
    :type tokens: list[str]
    :returns: Token list with matching date triplets replaced by ``"<date>"``.
    :rtype: list[str]
    """
    result = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            t0, t1, t2 = tokens[i], tokens[i + 1], tokens[i + 2]
            # day-first: "9th January 2018" or "9 January 2018"
            if (
                (_ORDINAL_RE.match(t0) or _DAY_NUM_RE.match(t0))
                and t1.lower() in _MONTHS
                and _YEAR_RE.match(t2)
            ):
                result.append("<date>")
                i += 3
                continue
            # month-first: "January 9th 2018" or "January 9 2018"
            if (
                t0.lower() in _MONTHS
                and (_ORDINAL_RE.match(t1) or _DAY_NUM_RE.match(t1))
                and _YEAR_RE.match(t2)
            ):
                result.append("<date>")
                i += 3
                continue
        result.append(tokens[i])
        i += 1
    return result
