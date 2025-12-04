# src/table_cleaning.py
from __future__ import annotations

import re
import pandas as pd


def _fix_spaced_letters(text: str) -> str:
    """
    Fix strings like 'C a b l e  A s s y Rotor Ch # 4'
    into 'Cable Assy Rotor Ch # 4'.
    """
    if not isinstance(text, str):
        return text

    s = text.strip()
    if not s:
        return s

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)

    tokens = s.split()
    if not tokens:
        return s

    # If average token length is small and we have enough tokens,
    # treat contiguous single letters as letter-spaced words.
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    if avg_len <= 1.5 and len(tokens) >= 4:
        grouped = []
        buffer = []

        for tok in tokens:
            if tok.isalpha() and len(tok) == 1:
                buffer.append(tok)
            else:
                if buffer:
                    grouped.append("".join(buffer))
                    buffer = []
                grouped.append(tok)

        if buffer:
            grouped.append("".join(buffer))

        s = " ".join(grouped)

    # Split CamelCase boundaries, e.g. CableAssy -> Cable Assy
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)

    # Final whitespace cleanup
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_table_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean spacing and letter-spaced text in all string columns of df.

    Uses *positional* indexing (iloc) so it works even when there are
    duplicate column names (where df[col] would return a DataFrame).
    """
    df = df.copy()
    n_cols = df.shape[1]

    for col_idx in range(n_cols):
        col_series = df.iloc[:, col_idx]
        if col_series.dtype == "object":
            df.iloc[:, col_idx] = col_series.apply(_fix_spaced_letters)

    return df