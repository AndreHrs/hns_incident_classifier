import re

def _get_tokenizer(
    keep_decimals_together: bool = True,
    split_num_units: bool = False,
):
    """Build and return a compiled regex tokenizer.

    Args:
        keep_decimals_together: If True, treat ``6.39`` as one token.
        split_num_units: If True, split ``10cm`` into ``["10", "cm"]``.

    Returns:
        Compiled regex pattern.
    """
    num_tok = r"(?i:<num>)"

    if split_num_units:
        if keep_decimals_together:
            pattern = (
                fr"{num_tok}"        # protect <num>
                r"|\d+\.\d+"         # decimal numbers
                r"|\d+"              # plain integers
                r"|[a-zA-Z°]+"
                r"|[^\w\s]"          # punctuation and symbols
            )
        else:
            pattern = (
                fr"{num_tok}"
                r"|\d+"
                r"|[a-zA-Z°]+"
                r"|[^\w\s]"
            )
    else:
        if keep_decimals_together:
            pattern = (
                fr"{num_tok}[a-zA-Z°]*"  # <num> followed by optional units
                fr"|{num_tok}"           # standalone <num>
                r"|\d+\.\d+"             # decimal numbers
                r"|\w+"
                r"|[^\w\s]"
            )
        else:
            pattern = (
                fr"{num_tok}[a-zA-Z°]*"
                fr"|{num_tok}"
                r"|\w+"
                r"|[^\w\s]"
            )
        return re.compile(pattern)
    return re.compile(pattern)


def handle_tokenization(
    text: str,
    keep_decimals_together: bool = True,
    split_num_units: bool = False,
) -> list[str]:
    """Tokenize a string into a list of tokens.

    Args:
        text: Input string to tokenize.
        keep_decimals_together: If True, treat ``6.39`` as one token rather
            than splitting on the decimal point.
        split_num_units: If True, split number-unit pairs like ``10cm``
            into ``["10", "cm"]``.

    Returns:
        List of string tokens.
    """
    regex = _get_tokenizer(keep_decimals_together, split_num_units)
    return regex.findall(text)
