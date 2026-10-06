#!/usr/bin/env python3
"""
Merge the book summaries and banned books datasets into a single CSV file.
Delete books until there are only the banned books and a control group of
non-banned books: one per banned book, sharing as many genres as possible and
with the closest publication date. No non-banned book is used twice.

Output: one row per book with the columns

    Author, Title, Publication Date, Genres, CMU Summary, Goodreads Summary, Banned Status

Books found in only the banned dataset (with no summary) will be removed.

"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from csv_preprocess import FINAL_COLUMNS as BANNED_COLUMNS
from csv_preprocess import _ascii_fold
from books_preprocess import FINAL_COLUMNS as SUMMARIES_COLUMNS
from books_preprocess import AUTHOR_THRESHOLD, GENRE_SEP, TITLE_THRESHOLD, match_books

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BANNED_PATH = Path("datasets/clean/bannedT_titles_clean.csv")
SUMMARIES_PATH = Path("datasets/clean/books_summaries_merged.csv")
OUTPUT_PATH = Path("datasets/clean/final_book_dataset.csv")

BANNED_COL = "Banned Status"
FINAL_COLUMNS = SUMMARIES_COLUMNS + [BANNED_COL]

# breaks ties between equally good control candidates
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_banned(path: Path = BANNED_PATH) -> pd.DataFrame:
    #PEN America clean CSV -> DataFrame[Author, Title] with authors as "First Last".
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df = df.dropna(subset=["Author", "Title"]).reset_index(drop=True)
    print(f"Banned     : {len(df)} books")
    return df


def load_summaries(path: Path = SUMMARIES_PATH) -> pd.DataFrame:
    #books_preprocess.py output -> DataFrame[SUMMARIES_COLUMNS].
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    print(f"Summaries  : {len(df)} books")
    return df


# --------------------------------------------------------------------------- #
# Fuzzy matching
# --------------------------------------------------------------------------- #
def match_banned(summaries: pd.DataFrame, banned: pd.DataFrame,
                 title_threshold: float = TITLE_THRESHOLD,
                 author_threshold: float = AUTHOR_THRESHOLD) -> pd.Series:

    #Return a boolean Series aligned with `summaries`: True if the book is in the banned list.
    #Banned books with no match in `summaries` are dropped (they have no summary).

    # match_books breaks ties on Goodreads popularity, which the merged CSV no longer has
    m = match_books(banned, summaries.assign(ratings_count=0),
                    title_threshold=title_threshold, author_threshold=author_threshold)

    is_banned = pd.Series(False, index=summaries.index)
    is_banned.iloc[m["gr_idx"]] = True

    print(f"Matched    : {is_banned.sum()} of {len(banned)} banned books have a summary")
    return is_banned


# --------------------------------------------------------------------------- #
# Control group
# --------------------------------------------------------------------------- #
def _genre_set(genres) -> frozenset:
    #"Genre; Genre" -> frozenset of lower-cased genres (empty if missing).
    if pd.isna(genres):
        return frozenset()
    return frozenset(g.strip().casefold() for g in str(genres).split(GENRE_SEP) if g.strip())


def _year(date):
    #"YYYY", "YYYY-MM" or "YYYY-MM-DD" -> int year, or None if missing/unparseable.
    m = re.match(r"\s*(\d{4})", "" if pd.isna(date) else str(date))
    return int(m.group(1)) if m else None


def _add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_genres"] = df["Genres"].map(_genre_set)
    df["_year"] = df["Publication Date"].map(_year).astype(float)
    return df


def _genre_bits(genre_sets: pd.Series) -> np.ndarray:
    #Series of genre sets -> (n, words) uint64 bitmask with one bit per distinct genre,
    #so genre overlap is a popcount of a bitwise AND.
    vocab, rows, cols = {}, [], []
    for row, genres in enumerate(genre_sets):
        for g in genres:
            rows.append(row)
            cols.append(vocab.setdefault(g, len(vocab)))
    rows, cols = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.uint64)
    bits = np.zeros((len(genre_sets), max(1, -(-len(vocab) // 64))), dtype=np.uint64)
    np.bitwise_or.at(bits, (rows, (cols // 64).astype(np.int64)), np.uint64(1) << (cols % 64))
    return bits


def _best_match(bits: np.ndarray, year: float, cand: dict):
    #Position in `cand` of the free candidate sharing the most genres with the book
    #(genre bitmask `bits`); ties are broken by the closest publication year.
    #Returns (position, shared genres), or (None, 0) if no candidate is free.
    # sample_control only passes candidates missing the same fields as the book,
    # so no genres -> overlap is 0 for all and date decides; no date -> distance
    # is equal for all and genres decide; neither -> first (random) candidate
    free = cand["free"]
    if not free.any():
        return None, 0
    overlap = np.bitwise_count(cand["bits"] & bits).sum(axis=1, dtype=np.int64)
    overlap = np.where(free, overlap, -1)
    best = overlap.max()
    distance = np.zeros(len(free)) if np.isnan(year) else np.abs(cand["year"] - year)
    distance = np.where(overlap == best, distance, np.inf)
    # candidates are pre-shuffled and argmin takes the first, so remaining ties are
    # random but reproducible
    return int(np.argmin(distance)), int(best)


# how a banned book is matched, keyed by (no_genres, no_year)
MATCH_MODES = {
    (False, False): "genres+date",
    (True, False): "date only",    # no genres -> control with no genres, closest date
    (False, True): "genres only",  # no date   -> control with no date, most genres
    (True, True): "random",        # neither   -> random control with neither
}


def _describe(book: pd.Series) -> str:
    year = "?" if pd.isna(book["_year"]) else int(book["_year"])
    return f"'{book['Title']}' ({year})"


def sample_control(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:

    #Keep every banned book plus one non-banned control book per banned book.
    #Each control is the unused non-banned book sharing the most genres with its
    #banned book, then the closest publication date. Controls are never reused.
    #A banned book missing genres and/or date is matched only against controls
    #missing the same fields (see MATCH_MODES); those matches are logged.

    df = _add_match_keys(df).sample(frac=1, random_state=seed)  # random tie-breaking
    bits = _genre_bits(df["_genres"])
    key = list(zip(df["_genres"].map(len).eq(0), df["_year"].isna()))
    is_banned = df[BANNED_COL].to_numpy(dtype=bool)

    # candidate pools, one per missing-fields key
    pools = {}
    for k in MATCH_MODES:
        pos = np.flatnonzero(~is_banned & np.array([x == k for x in key]))
        pools[k] = {"pos": pos, "bits": bits[pos], "year": df["_year"].to_numpy()[pos],
                    "free": np.ones(len(pos), dtype=bool)}

    # greedy matching is order dependent: books with the most genres pick first
    banned_pos = np.flatnonzero(is_banned)
    n_genres = df["_genres"].map(len).to_numpy()[banned_pos]
    banned_pos = banned_pos[np.argsort(-n_genres, kind="stable")]

    control_pos, counts, unmatched = [], {mode: 0 for mode in MATCH_MODES.values()}, 0
    weak = 0  # genres+date matches that share no genre
    for b in tqdm(banned_pos, desc="Control", leave=False):
        book, mode, cand = df.iloc[b], MATCH_MODES[key[b]], pools[key[b]]
        match, shared = _best_match(bits[b], book["_year"], cand)
        if match is None:
            unmatched += 1
            tqdm.write(f"  [{mode}] no control left for {_describe(book)}")
            continue

        cand["free"][match] = False  # no repeats
        control_pos.append(cand["pos"][match])
        counts[mode] += 1
        if mode == "genres+date" and shared == 0:
            weak += 1
        if mode != "genres+date":
            tqdm.write(f"  [{mode}] {_describe(book)} -> {_describe(df.iloc[control_pos[-1]])}")

    banned, control = df.iloc[banned_pos], df.iloc[control_pos]

    print(f"Control    : {len(control)} non-banned books matched to {len(banned)} banned")
    for mode, n in counts.items():
        print(f"             {n:>5} by {mode}")
    if weak:
        print(f"             {weak:>5} of the genres+date matches share no genre")
    if unmatched:
        print(f"             {unmatched:>5} banned books left without a control")
    return pd.concat([banned, control], ignore_index=True)


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def merge_datasets(summaries: pd.DataFrame, banned: pd.DataFrame,
                   seed: int = RANDOM_SEED, **thresholds) -> pd.DataFrame:
    summaries = summaries.copy()
    summaries[BANNED_COL] = match_banned(summaries, banned, **thresholds)

    out = sample_control(summaries, seed=seed).reindex(columns=FINAL_COLUMNS)
    out = out.sort_values(
        ["Author", "Title"],
        key=lambda s: s.map(lambda v: _ascii_fold(str(v)).casefold()),
    ).reset_index(drop=True)

    print(f"Output rows: {len(out)} ({out[BANNED_COL].sum()} banned)")
    return out


def main() -> None:
    # titles in the match log can be in any script; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Merge book summaries with the banned books list.")
    ap.add_argument("--banned", type=Path, default=BANNED_PATH)
    ap.add_argument("--summaries", type=Path, default=SUMMARIES_PATH)
    ap.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--title-threshold", type=float, default=TITLE_THRESHOLD)
    ap.add_argument("--author-threshold", type=float, default=AUTHOR_THRESHOLD)
    args = ap.parse_args()

    banned = load_banned(args.banned)
    summaries = load_summaries(args.summaries)
    out = merge_datasets(summaries, banned, seed=args.seed,
                         title_threshold=args.title_threshold,
                         author_threshold=args.author_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
