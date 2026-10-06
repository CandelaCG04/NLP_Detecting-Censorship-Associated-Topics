#!/usr/bin/env python3
"""
Clean the CMU Book Summary dataset and the Goodreads (UCSD Book Graph) books
dataset, then merge them by fuzzy-matching (author, title) pairs.

Output: one row per book with the columns

    Author, Title, Publication Date, Genres, CMU Summary, Goodreads Summary

Books found in only one dataset keep an empty value for the other summary.
Publication Date and Genres come from CMU when available, else from Goodreads
(Goodreads dates are those of the most-rated edition, not the original work).

"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from csv_preprocess import NAME_SUFFIXES, _ascii_fold, author_key, clean_text, title_key

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CMU_PATH = Path("datasets/raw/books/booksummaries.txt")
GOODREADS_PATH = Path("datasets/raw/books/goodreads_books.json")
GOODREADS_AUTHORS_PATH = Path("datasets/raw/books/goodreads_book_authors.json.gz")
GOODREADS_GENRES_PATH = Path("datasets/raw/books/goodreads_book_genres_initial.json.gz")
GOODREADS_CACHE = Path("datasets/interim/goodreads_clean.pkl")
OUTPUT_PATH = Path("datasets/clean/books_summaries_merged.csv")

CMU_COLUMNS = ["Wikipedia ID", "Freebase ID", "Title", "Author",
               "Publication Date", "Genres", "Summary"]
FINAL_COLUMNS = ["Author", "Title", "Publication Date", "Genres",
                 "CMU Summary", "Goodreads Summary"]
GENRE_SEP = "; "

# rapidfuzz scores (0-100) a candidate pair must reach to count as the same book
TITLE_THRESHOLD = 90
AUTHOR_THRESHOLD = 85

# Getting read of subtitles and series info in titles 
SUBTITLE_RE = re.compile(r"\s*[:;]\s.*$|\s+-\s.*$")


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def _drop_incomplete(df: pd.DataFrame, summary_col: str) -> pd.DataFrame:
    #Clean text columns and drop rows missing author, title or summary.
    for col in ("Author", summary_col):
        df[col] = df[col].map(clean_text)
    df["Title"] = df["Title"].map(lambda v: clean_text(v, placeholders=False))
    return df.dropna(subset=["Author", "Title", summary_col]).reset_index(drop=True)


def _cmu_genres(value):
    #Freebase JSON dict {"/m/..": "Genre", ...} -> "Genre; Genre".
    if not isinstance(value, str):
        return None
    return GENRE_SEP.join(json.loads(value).values()) or None


def _goodreads_date(book: dict):
    #publication_year/month/day -> "YYYY", "YYYY-MM" or "YYYY-MM-DD" (CMU format).
    parts = []
    for key, width in (("publication_year", 4), ("publication_month", 2), ("publication_day", 2)):
        v = book.get(key)
        if not v or not str(v).isdigit():
            break
        parts.append(str(int(v)).zfill(width))
    return "-".join(parts) or None


def load_cmu(path: Path = CMU_PATH) -> pd.DataFrame:
    #CMU Book Summary Dataset -> DataFrame[Author, Title, Summary, Publication Date, Genres].
    df = pd.read_csv(path, sep="\t", header=None, names=CMU_COLUMNS, dtype=str,
                     quoting=csv.QUOTE_NONE, keep_default_na=False, na_values=[""],
                     encoding="utf-8")
    n_raw = len(df)
    df["Genres"] = df["Genres"].map(_cmu_genres)
    df = _drop_incomplete(df[["Author", "Title", "Summary", "Publication Date", "Genres"]].copy(),
                          "Summary")
    df = df.drop_duplicates(subset=["Author", "Title"]).reset_index(drop=True)
    print(f"CMU        : {n_raw} raw -> {len(df)} with author, title and summary")
    return df


def load_goodreads_authors(path: Path = GOODREADS_AUTHORS_PATH) -> dict[str, str]:
    #author_id -> author name.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return {a["author_id"]: a["name"] for a in map(json.loads, f)}


def _primary_author(authors: list[dict], names: dict[str, str]):
    #First credited author, preferring entries without a role (translator, illustrator...).
    ordered = [a for a in authors if not a.get("role")] + [a for a in authors if a.get("role")]
    for a in ordered:
        name = names.get(a.get("author_id"))
        if name:
            return name
    return None


def load_goodreads_genres(path: Path = GOODREADS_GENRES_PATH) -> dict[str, str]:
    #book_id -> "genre; genre", most-voted genre first.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return {g["book_id"]: GENRE_SEP.join(sorted(g["genres"], key=g["genres"].get, reverse=True))
                for g in map(json.loads, f) if g["genres"]}


def _add_goodreads_genres(df: pd.DataFrame, genres_path: Path | None) -> pd.DataFrame:
    if genres_path is None or not genres_path.exists():
        print(f"Goodreads  : genres file {genres_path} not found, Goodreads genres left empty")
        df["Genres"] = None
    else:
        df["Genres"] = df["book_id"].map(load_goodreads_genres(genres_path))
    return df.drop(columns="book_id")


def load_goodreads(path: Path = GOODREADS_PATH,
                   authors_path: Path = GOODREADS_AUTHORS_PATH,
                   cache: Path | None = GOODREADS_CACHE,
                   genres_path: Path | None = GOODREADS_GENRES_PATH) -> pd.DataFrame:

    #Goodreads books (JSON lines) -> DataFrame[Author, Title, Summary, ratings_count, Publication Date, Genres].
    if cache is not None and cache.exists():
        df = pd.read_pickle(cache)
        if {"book_id", "Publication Date"} <= set(df.columns):
            print(f"Goodreads  : {len(df)} works loaded from cache {cache}")
            return _add_goodreads_genres(df, genres_path)
        print(f"Goodreads  : cache {cache} is outdated, re-parsing")

    names = load_goodreads_authors(authors_path)
    rows, n_raw = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading Goodreads", unit=" books"):
            n_raw += 1
            if '"description": ""' in line:          # cheap skip before parsing
                continue
            book = json.loads(line)
            rows.append((
                book.get("work_id") or book.get("book_id"),
                book.get("book_id"),
                _primary_author(book.get("authors", []), names),
                book.get("title_without_series") or book.get("title"),
                book.get("description"),
                int(book.get("ratings_count") or 0),
                _goodreads_date(book),
            ))

    df = pd.DataFrame(rows, columns=["work_id", "book_id", "Author", "Title", "Summary",
                                     "ratings_count", "Publication Date"])
    df = _drop_incomplete(df, "Summary")
    df = (df.sort_values("ratings_count", ascending=False, kind="stable")
            .drop_duplicates(subset="work_id")
            .drop(columns="work_id")
            .reset_index(drop=True))
    print(f"Goodreads  : {n_raw} raw -> {len(df)} works with author, title and summary")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return _add_goodreads_genres(df, genres_path)


# --------------------------------------------------------------------------- #
# Fuzzy matching
# --------------------------------------------------------------------------- #
def _surname(author: str) -> str:
    tokens = [t for t in re.sub(r"[^a-z0-9\s]", " ", _ascii_fold(author).lower()).split()
              if t not in NAME_SUFFIXES]
    return tokens[-1] if tokens else ""


def _add_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_author_key"] = df["Author"].map(author_key)
    df["_surname"] = df["Author"].map(_surname)
    df["_title_key"] = df["Title"].map(title_key)
    df["_short_title_key"] = df["Title"].map(lambda t: title_key(SUBTITLE_RE.sub("", t) or t))
    return df


def _title_score(a_full: str, a_short: str, b_full: str, b_short: str) -> float:
    # 'Vol. 1' vs 'Vol. 2' are nearly identical strings but different books
    if re.findall(r"\d+", a_short) != re.findall(r"\d+", b_short):
        return 0.0
    return max(fuzz.ratio(a_full, b_full), fuzz.ratio(a_short, b_short))


def match_books(cmu: pd.DataFrame, goodreads: pd.DataFrame,
                title_threshold: float = TITLE_THRESHOLD,
                author_threshold: float = AUTHOR_THRESHOLD) -> pd.DataFrame:

    #Return DataFrame[cmu_idx, gr_idx, title_score, author_score] with at most one Goodreads work per CMU book and vice versa.

    cmu, gr = _add_match_keys(cmu), _add_match_keys(goodreads)

    by_surname, by_title = defaultdict(list), defaultdict(list)
    for i, (s, t) in enumerate(zip(gr["_surname"], gr["_short_title_key"])):
        by_surname[s].append(i)
        by_title[t].append(i)

    gr_author, gr_title = gr["_author_key"].tolist(), gr["_title_key"].tolist()
    gr_short, gr_ratings = gr["_short_title_key"].tolist(), gr["ratings_count"].tolist()

    matches = []
    cmu_rows = zip(cmu["_author_key"], cmu["_surname"], cmu["_title_key"], cmu["_short_title_key"])
    for ci, (author, surname, title, short) in tqdm(enumerate(cmu_rows), total=len(cmu),
                                                    desc="Matching", unit=" books"):
        candidates = set(by_surname.get(surname, ())) | set(by_title.get(short, ()))
        best, best_key = None, None
        for gi in candidates:
            t_score = _title_score(title, short, gr_title[gi], gr_short[gi])
            if t_score < title_threshold:
                continue
            a_score = fuzz.token_sort_ratio(author, gr_author[gi])
            if a_score < author_threshold:
                continue
            key = (t_score + a_score, gr_ratings[gi])
            if best_key is None or key > best_key:
                best, best_key = (ci, gi, t_score, a_score), key
        if best is not None:
            matches.append(best)

    m = pd.DataFrame(matches, columns=["cmu_idx", "gr_idx", "title_score", "author_score"])
    # a Goodreads work may be the best candidate for several CMU rows: keep the closest
    m["_total"] = m["title_score"] + m["author_score"]
    m = (m.sort_values("_total", ascending=False, kind="stable")
          .drop_duplicates(subset="gr_idx")
          .drop(columns="_total")
          .sort_values("cmu_idx")
          .reset_index(drop=True))
    return m


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def merge_datasets(cmu: pd.DataFrame, goodreads: pd.DataFrame,
                   matched_only: bool = False, **thresholds) -> pd.DataFrame:
    m = match_books(cmu, goodreads, **thresholds)

    cmu_m = cmu.iloc[m["cmu_idx"]].reset_index(drop=True)
    gr_m = goodreads.iloc[m["gr_idx"]].reset_index(drop=True)
    matched = pd.DataFrame({
        "Author": cmu_m["Author"],
        "Title": cmu_m["Title"],
        # CMU metadata is per work; Goodreads only fills the gaps
        "Publication Date": cmu_m["Publication Date"].fillna(gr_m["Publication Date"]),
        "Genres": cmu_m["Genres"].fillna(gr_m["Genres"]),
        "CMU Summary": cmu_m["Summary"],
        "Goodreads Summary": gr_m["Summary"],
    })
    parts = [matched]
    if not matched_only:
        cmu_only = cmu.drop(index=m["cmu_idx"]).rename(columns={"Summary": "CMU Summary"})
        gr_only = goodreads.drop(index=m["gr_idx"]).rename(columns={"Summary": "Goodreads Summary"})
        parts += [cmu_only, gr_only]

    out = pd.concat(parts, ignore_index=True).reindex(columns=FINAL_COLUMNS)
    out = out.sort_values(
        ["Author", "Title"],
        key=lambda s: s.map(lambda v: _ascii_fold(str(v)).casefold()),
    ).reset_index(drop=True)

    print(f"Matched    : {len(m)} books have both summaries "
          f"({len(m) / len(cmu):.1%} of CMU)")
    print(f"Output rows: {len(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge CMU and Goodreads book summaries.")
    ap.add_argument("--cmu", type=Path, default=CMU_PATH)
    ap.add_argument("--goodreads", type=Path, default=GOODREADS_PATH)
    ap.add_argument("--goodreads-authors", type=Path, default=GOODREADS_AUTHORS_PATH)
    ap.add_argument("--goodreads-genres", type=Path, default=GOODREADS_GENRES_PATH)
    ap.add_argument("--cache", type=Path, default=GOODREADS_CACHE,
                    help="pickle of the cleaned Goodreads data (delete it to re-parse)")
    ap.add_argument("-o", "--output", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--matched-only", action="store_true",
                    help="only keep books present in both datasets")
    ap.add_argument("--title-threshold", type=float, default=TITLE_THRESHOLD)
    ap.add_argument("--author-threshold", type=float, default=AUTHOR_THRESHOLD)
    args = ap.parse_args()

    cmu = load_cmu(args.cmu)
    goodreads = load_goodreads(args.goodreads, args.goodreads_authors, args.cache,
                               args.goodreads_genres)
    out = merge_datasets(cmu, goodreads, matched_only=args.matched_only,
                         title_threshold=args.title_threshold,
                         author_threshold=args.author_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
