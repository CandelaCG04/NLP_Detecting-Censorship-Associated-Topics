#!/usr/bin/env python3
"""
Print descriptive statistics of the final book dataset, comparing banned books
against the non-banned control group:

    - summary source   : CMU only, Goodreads only or both
    - publication date : missing dates, year distribution and decades
    - genres           : missing genres, genres per book and most common genres

Publication Date and Genres come from CMU when the book has a CMU summary, else
from Goodreads (see books_preprocess.py), so the two genre vocabularies differ
("Science Fiction" vs "fantasy, paranormal"). Genre counts are therefore also
broken down by the source they come from.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# allow running as a file (python scripts/x.py) as well as a module (python -m scripts.x)
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge import BANNED_COL, OUTPUT_PATH as DATASET_PATH
from scripts.merge import _genre_set, _year

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
GROUPS = {True: "Banned", False: "Not banned"}
SOURCES = ["CMU only", "Goodreads only", "Both", "None"]
TOP_GENRES = 15


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    #merge.py output -> DataFrame with helper columns _group, _source, _year, _genres.
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    df[BANNED_COL] = df[BANNED_COL].str.casefold().eq("true")
    df["_group"] = df[BANNED_COL].map(GROUPS)
    df["_source"] = summary_source(df)
    df["_year"] = df["Publication Date"].map(_year).astype(float)
    df["_genres"] = df["Genres"].map(_genre_set)
    print(f"Dataset    : {len(df)} books ({df[BANNED_COL].sum()} banned)")
    return df


def summary_source(df: pd.DataFrame) -> pd.Series:
    #Which summaries each book has: "CMU only", "Goodreads only", "Both" or "None".
    cmu, gr = df["CMU Summary"].notna(), df["Goodreads Summary"].notna()
    source = pd.Series("None", index=df.index)
    source[cmu & ~gr] = "CMU only"
    source[~cmu & gr] = "Goodreads only"
    source[cmu & gr] = "Both"
    return source


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _header(title: str) -> None:
    print(f"\n{'=' * 79}\n{title}\n{'=' * 79}")


def _count_pct(df: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    #Count and within-group percentage of each value of `col`, one column pair per group.
    counts = pd.crosstab(df[col], df["_group"]).reindex(columns=list(GROUPS.values()), fill_value=0)
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    pct = counts / counts.sum() * 100
    out = pd.concat({"n": counts, "%": pct.round(1)}, axis=1).swaplevel(axis=1)
    return out.reindex(columns=list(GROUPS.values()), level=0)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def source_stats(df: pd.DataFrame) -> pd.DataFrame:
    #Number and share of books per summary source, banned vs not banned.
    _header("Summary source")
    out = _count_pct(df, "_source", order=SOURCES)
    print(out.to_string())
    return out


def date_stats(df: pd.DataFrame) -> pd.DataFrame:
    #Missing dates and publication year distribution, banned vs not banned.
    _header("Publication date")
    missing = df.groupby("_group")["_year"].apply(lambda s: s.isna().sum())
    out = df.groupby("_group")["_year"].describe(percentiles=[.1, .25, .5, .75, .9])
    out.insert(0, "missing", missing)
    out = out.reindex(GROUPS.values()).round(1)
    print(out.to_string())

    # absolute year difference is small by construction (control is date matched)
    print("\nBy decade (% of books with a date):")
    dated = df[df["_year"].notna()]
    # older decades are sparse; group them so the table stays readable
    decade = dated["_year"].map(lambda y: "<1950" if y < 1950 else f"{int(y) // 10 * 10}s")
    dated = dated.assign(_decade=decade)
    order = sorted(decade.unique(), key=lambda d: (d != "<1950", d))
    print(_count_pct(dated, "_decade", order=order).to_string())
    return out


def genre_stats(df: pd.DataFrame, top: int = TOP_GENRES) -> pd.DataFrame:
    #Missing genres, genres per book and most common genres, banned vs not banned.
    _header("Genres")
    n_genres = df["_genres"].map(len)
    out = n_genres.groupby(df["_group"]).describe()[["mean", "50%", "max"]]
    out.insert(0, "missing", n_genres.eq(0).groupby(df["_group"]).sum())
    out = out.reindex(GROUPS.values()).round(2)
    print("Genres per book:")
    print(out.to_string())

    # dates and genres come from CMU when the book has a CMU summary
    genre_source = df["CMU Summary"].notna().map({True: "CMU", False: "Goodreads"})
    print("\nGenre source (% of books):")
    print(_count_pct(df.assign(_genre_source=genre_source), "_genre_source",
                     order=["CMU", "Goodreads"]).to_string())

    for name in ["CMU", "Goodreads"]:
        print(f"\nTop {top} {name} genres (% of books in the group with {name} genres):")
        print(top_genres(df[genre_source == name], top).to_string())
    return out


def top_genres(df: pd.DataFrame, top: int = TOP_GENRES) -> pd.DataFrame:
    #Share of books in each group tagged with each genre, for the `top` most common
    #genres overall; "ratio" is banned share / not banned share (>1: more common in banned).
    df = df[df["_genres"].map(len) > 0]
    exploded = df[["_group", "_genres"]].explode("_genres", ignore_index=True)
    counts = pd.crosstab(exploded["_genres"], exploded["_group"])
    counts = counts.reindex(columns=list(GROUPS.values()), fill_value=0)
    pct = counts / df["_group"].value_counts().reindex(counts.columns) * 100

    out = pct.round(1)
    out["ratio"] = (pct["Banned"] / pct["Not banned"]).round(2)
    out = out.loc[counts.sum(axis=1).sort_values(ascending=False).index[:top]]
    out.index.name = None
    out.columns.name = None
    return out


def dataset_stats(df: pd.DataFrame, top: int = TOP_GENRES) -> dict[str, pd.DataFrame]:
    return {
        "source": source_stats(df),
        "date": date_stats(df),
        "genres": genre_stats(df, top),
    }


def main() -> None:
    # titles and genres can be in any script; Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Compare banned and non-banned books in the final dataset.")
    ap.add_argument("-i", "--input", type=Path, default=DATASET_PATH)
    ap.add_argument("--top", type=int, default=TOP_GENRES, help="number of genres to list")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    df = load_dataset(args.input)
    dataset_stats(df, top=args.top)


if __name__ == "__main__":
    main()
