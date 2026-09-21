#!/usr/bin/env python3
"""
Preprocess the PEN America Index of School Book Bans (2021-2025).

Reads every yearly CSV (the schema changes between years), harmonises the
columns, cleans/normalises the text, parses the dates, and collapses everything
to ONE ROW PER UNIQUE BOOK with the columns:

    Author, Title, Type of Ban, Secondary Author(s),
    Date of Challenge/Removal, Ban Status

Usage
-----
    python preprocess_pen_bans.py data/pen_bans/ -o banned_titles_clean.csv
    python preprocess_pen_bans.py 2022.csv 2023.csv 2025.csv --flip-authors

Or from Python:
    from preprocess_pen_bans import preprocess
    df = preprocess(["a.csv", "b.csv"])
"""
from __future__ import annotations

import argparse
import re
import string
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATE_COL = "Date of Challenge/Removal"
FINAL_COLUMNS = [
    "Author",
    "Title",
    "Type of Ban",
    "Secondary Author(s)",
    DATE_COL,
    "Ban Status",
]

# Header variants seen across years -> canonical name (keys are lower-cased).
COLUMN_ALIASES = {
    "author": "Author",
    "authors": "Author",
    "title": "Title",
    "type of ban": "Type of Ban",
    "ban status": "Ban Status",
    "secondary author(s)": "Secondary Author(s)",
    "secondary authors": "Secondary Author(s)",
    "secondary author": "Secondary Author(s)",
    "date of challenge/removal": DATE_COL,
    "date of challenge / removal": DATE_COL,
    "date of challenge or removal": DATE_COL,
}

# Strings that really mean "missing" (NOT applied to titles).
PLACEHOLDERS = {"", "n/a", "na", "none", "null", "nan", "-", "--", "unknown"}

# Add entries here after inspecting the label counts printed at the end, e.g.
# {"Banned In Libraries And Classrooms": "Banned from Libraries and Classrooms"}
BAN_LABEL_MAP: dict[str, str] = {}

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
SMALL_WORDS = {"and", "or", "from", "in", "of", "the", "to", "for", "on", "at"}

# Explicit formats only. Never let a generic parser guess: "Apr-22" would be
# read as "April 22nd of the current year" instead of "April 2022".
DATE_FORMATS = (
    "%b-%y", "%B-%y", "%b %y", "%B %y",
    "%b-%Y", "%B-%Y", "%b %Y", "%B %Y",
    "%Y-%m", "%m/%Y", "%Y-%m-%d", "%m/%d/%Y",
)

# "Sloppy Firsts (Jessica Darling Series)" -> "Sloppy Firsts"
SERIES_PAREN_RE = re.compile(
    r"\s*[\(\[][^()\[\]]*\b(?:series|trilogy|saga|quartet)\b[^()\[\]]*[\)\]]\s*$",
    re.IGNORECASE,
)

_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _norm_header(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).replace("\ufeff", "").strip().lower())


def load_csv(path: Path) -> pd.DataFrame:
    """Read one yearly file and return it with the canonical columns only."""
    df = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            # keep_default_na=False so a book titled "Null" or "None" survives
            df = pd.read_csv(path, dtype=str, encoding=enc,
                             keep_default_na=False, na_values=[""])
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"Could not decode {path}")

    df.columns = [COLUMN_ALIASES.get(_norm_header(c), str(c).strip()) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]          # keep first if header repeats
    if "Title" not in df.columns:
        raise ValueError(f"{path.name}: no 'Title' column found. Headers: {list(df.columns)}")
    for col in FINAL_COLUMNS:                         # e.g. "Ban Status" missing in 2021-22
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FINAL_COLUMNS].copy()
    df["_source"] = path.name
    return df


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def clean_text(value, placeholders: bool = True):
    """Unicode-normalise, collapse whitespace, drop wrapping quotes."""
    if pd.isna(value):
        return pd.NA
    s = unicodedata.normalize("NFKC", str(value)).translate(_PUNCT_MAP)
    s = re.sub(r"[\u200b\ufeff]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 1 and s[0] == s[-1] == '"':
        s = s[1:-1].strip()
    if not s or (placeholders and s.lower() in PLACEHOLDERS):
        return pd.NA
    return s


def fix_case(s):
    """Only touch strings that are entirely UPPER or lower case."""
    if pd.isna(s):
        return s
    return string.capwords(s) if (s.isupper() or s.islower()) else s


def fix_inverted_article(title):
    if pd.isna(title):
        return title
    m = re.fullmatch(r"(.+?),\s*(The|A|An)", title, flags=re.IGNORECASE)
    return f"{m.group(2).capitalize()} {m.group(1)}" if m else title


def strip_series(title):
    if pd.isna(title):
        return title
    stripped = SERIES_PAREN_RE.sub("", title).strip()
    return stripped or title


def flip_name(name):
    """'McCafferty, Megan' -> 'Megan McCafferty' (also handles ', Jr.')."""
    if pd.isna(name) or "," not in name:
        return name
    parts = [p.strip() for p in name.split(",") if p.strip()]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    if len(parts) == 3 and parts[2].lower().rstrip(".") in NAME_SUFFIXES:
        return f"{parts[1]} {parts[0]} {parts[2]}"
    return name


def normalize_ban_label(value):
    """Consistent casing/spacing for 'Type of Ban' and 'Ban Status'."""
    if pd.isna(value):
        return value
    words = value.lower().split()
    s = " ".join(w if (i and w in SMALL_WORDS) else w.capitalize()
                 for i, w in enumerate(words))
    return BAN_LABEL_MAP.get(s, s)


def parse_month(value):
    """'Apr-22' / 'June 2025' / 'Sept 2023' -> Timestamp(first day of that month)."""
    if pd.isna(value):
        return pd.NaT
    s = re.sub(r"\bSept\b", "Sep", str(value).replace(".", "").strip(), flags=re.I)
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt)).replace(day=1)
        except ValueError:
            continue
    return pd.NaT


# --------------------------------------------------------------------------- #
# Dedup keys
# --------------------------------------------------------------------------- #
def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def title_key(title) -> str:
    t = _ascii_fold(str(title)).lower().replace("&", " and ")
    t = re.sub(r"['`]", "", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t or str(title).lower()


def author_key(author) -> str:
    """Order-insensitive: 'Rowling, J.K.' == 'J. K. Rowling'."""
    if pd.isna(author):
        return ""
    a = re.sub(r"[^a-z0-9\s]", " ", _ascii_fold(str(author)).lower())
    return " ".join(sorted(t for t in a.split() if t not in NAME_SUFFIXES))


def _similar(a: str, b: str, threshold: float) -> bool:
    # 'Vol. 1' vs 'Vol. 2' are nearly identical strings but different books
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


def merge_similar_titles(df: pd.DataFrame, threshold: float) -> int:
    """Fuzzy-merge title keys *within the same author* (typos, subtitle drift)."""
    mapping: dict[tuple[str, str], str] = {}
    for a_key, block in df.groupby("_author_key", sort=False):
        reps: list[str] = []
        for t in block["_title_key"].unique():
            if a_key == "":                      # no author -> too risky to fuzzy-match
                mapping[(a_key, t)] = t
                continue
            match = next((r for r in reps if _similar(t, r, threshold)), None)
            if match is None:
                reps.append(t)
                match = t
            mapping[(a_key, t)] = match
    df["_title_key"] = [mapping[(a, t)] for a, t in zip(df["_author_key"], df["_title_key"])]
    return sum(1 for (_, t), v in mapping.items() if t != v)


# --------------------------------------------------------------------------- #
# Aggregation helpers (one row per book)
# --------------------------------------------------------------------------- #
def _mode(s: pd.Series):
    """Most frequent value; on a tie prefer the longer (more complete) one."""
    counts = s.dropna().value_counts()
    if counts.empty:
        return pd.NA
    return max(counts.index, key=lambda v: (counts[v], len(v)))


def _join_unique(s: pd.Series, sep: str = " | "):
    seen: list[str] = []
    for v in s.dropna():
        if v not in seen:
            seen.append(v)
    return sep.join(seen) if seen else pd.NA


def _union_secondary(s: pd.Series):
    seen: dict[str, str] = {}
    for v in s.dropna():
        for part in re.split(r"\s*[;|]\s*", v):
            if part.strip():
                seen.setdefault(part.strip().casefold(), part.strip())
    return "; ".join(seen.values()) if seen else pd.NA


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def preprocess(paths, flip_authors: bool = False, fuzzy_threshold: float = 0.92) -> pd.DataFrame:
    paths = [Path(p) for p in paths]
    df = pd.concat([load_csv(p) for p in paths], ignore_index=True)
    n_raw = len(df)

    # --- clean text columns --------------------------------------------------
    for col in ("Author", "Secondary Author(s)", "Type of Ban", "Ban Status"):
        df[col] = df[col].map(clean_text)
    df["Title"] = df["Title"].map(lambda v: clean_text(v, placeholders=False))
    df = df.dropna(subset=["Title"]).copy()

    df["Title"] = df["Title"].map(strip_series).map(fix_inverted_article).map(fix_case)
    df["Author"] = df["Author"].map(fix_case)
    if flip_authors:
        df["Author"] = df["Author"].map(flip_name)
    for col in ("Type of Ban", "Ban Status"):
        df[col] = df[col].map(normalize_ban_label)

    # --- dates ---------------------------------------------------------------
    raw_dates = df[DATE_COL].map(clean_text)
    parsed = raw_dates.map(parse_month)
    unparsed = raw_dates[raw_dates.notna() & parsed.isna()].unique()
    if len(unparsed):
        print(f"[warn] {len(unparsed)} unparsed date value(s): {list(unparsed)[:10]}")
    df[DATE_COL] = pd.to_datetime(parsed)

    # --- dedup keys ----------------------------------------------------------
    df["_author_key"] = df["Author"].map(author_key)
    df["_title_key"] = df["Title"].map(title_key)
    n_fuzzy = merge_similar_titles(df, fuzzy_threshold) if fuzzy_threshold < 1 else 0

    # chronological order so joined labels read oldest -> newest
    df = df.sort_values(DATE_COL, na_position="last", kind="stable")

    out = (
        df.groupby(["_author_key", "_title_key"], sort=False)
          .agg(**{
              "Author": ("Author", _mode),
              "Title": ("Title", _mode),
              "Type of Ban": ("Type of Ban", _join_unique),
              "Secondary Author(s)": ("Secondary Author(s)", _union_secondary),
              DATE_COL: (DATE_COL, "min"),            # earliest challenge
              "Ban Status": ("Ban Status", _join_unique),
          })
          .reset_index(drop=True)[FINAL_COLUMNS]
    )
    out = out.sort_values(
        ["Author", "Title"],
        key=lambda s: s.map(lambda v: _ascii_fold(str(v)).casefold() if pd.notna(v) else v),
        na_position="last",
    ).reset_index(drop=True)

    # --- report --------------------------------------------------------------
    print(f"Files read           : {len(paths)}")
    print(f"Raw ban records      : {n_raw}")
    print(f"Unique titles        : {len(out)}  (fuzzy-merged {n_fuzzy} title variants)")
    print(f"Missing author       : {out['Author'].isna().sum()}")
    print(f"Missing date         : {out[DATE_COL].isna().sum()}")
    for col in ("Type of Ban", "Ban Status"):
        print(f"\nDistinct '{col}' values:")
        print(out[col].value_counts(dropna=False).head(15).to_string())
    return out


def _collect_paths(inputs, exclude: Path) -> list[Path]:
    paths: list[Path] = []
    for item in map(Path, inputs):
        found = sorted(item.glob("*.csv")) if item.is_dir() else [item]
        paths += [p for p in found if p.resolve() != exclude.resolve()]
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess PEN America book-ban CSVs.")
    ap.add_argument("inputs", nargs="+", help="CSV files and/or folders containing CSVs")
    ap.add_argument("-o", "--output", default="banned_titles_clean.csv")
    ap.add_argument("--flip-authors", action="store_true",
                    help="'Last, First' -> 'First Last' (matches Goodreads/CMU/Open Library)")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.92,
                    help="title similarity (0-1) for merging variants within an author; 1 disables")
    args = ap.parse_args()

    out_path = Path(args.output)
    paths = _collect_paths(args.inputs, out_path)
    if not paths:
        raise SystemExit("No CSV files found.")

    df = preprocess(paths, flip_authors=args.flip_authors, fuzzy_threshold=args.fuzzy_threshold)
    df.to_csv(out_path, index=False, date_format="%Y-%m")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()








#!!!!!!!!!!!1
#!!!!!!!!!!!

# import pandas as pd
# import os
# import glob

# def load_csv_files(directory="/datasets_raw/banned", pattern="*.csv"):
#     """
#     Load all CSV files matching a pattern from a directory into DataFrames.
    
#     Returns:
#         dict: {filename: DataFrame}
#     """
#     dataframes = {}
#     csv_files = glob.glob(os.path.join(directory, pattern))
    
#     if not csv_files:
#         print(f"No CSV files found in '{directory}' matching '{pattern}'")
#         return dataframes
    
#     for filepath in csv_files:
#         filename = os.path.basename(filepath)
#         try:
#             df = pd.read_csv(filepath, encoding="utf-8")
#             dataframes[filename] = df
#             print(f"✓ Loaded '{filename}': {df.shape[0]} rows, {df.shape[1]} columns")
#         except Exception as e:
#             print(f"✗ Failed to load '{filename}': {e}")
    
#     return dataframes


# def load_single_csv(filepath):
#     """Load a single CSV file into a DataFrame."""
#     df = pd.read_csv(filepath, encoding="utf-8")
    
#     # Optional: parse the date column if present
#     date_col = "Date of Challenge/Removal"
#     if date_col in df.columns:
#         df[date_col] = pd.to_datetime(df[date_col], format="%b-%y", errors="coerce")
    
#     return df


# if __name__ == "__main__":
#     # Option 1: Load a single file
#     df = load_single_csv("./datasets_raw/banned/BannedBooks2021-2022.csv")
#     print(df.head())
#     print(df.info())
    
#     # Option 2: Load all CSVs in a folder
#     #dfs = load_csv_files(directory=".", pattern="*.csv")
    
#     # Example: work with the first DataFrame
#     # if df:
#     #     print("\n--- Preview ---")
#     #     print(df.head())
#     #     print("\n--- Columns ---")
#     #     print(list(df.columns))
#     #     print("\n--- Dtypes ---")
#     #     print(df.dtypes)


# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1111
# #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!