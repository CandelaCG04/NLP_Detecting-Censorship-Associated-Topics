#!/usr/bin/env python3
"""
Build one clean text per book for topic modelling and split it into train/test.

Text is the Goodreads summary, falling back to the CMU summary. To prevent
label leakage, character names (spaCy PERSON entities), the book's own title,
sentences about bans/awards/bestseller lists and quoted review blurbs are
removed. Non-English, very short and duplicate texts are dropped.

Output: two CSVs (train/test, stratified on Banned Status) with the columns

    Book ID, Banned Status, Text

"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import spacy
from sklearn.model_selection import train_test_split


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.csv_preprocess import _ascii_fold
from scripts.merge import BANNED_COL, ID_COL, OUTPUT_PATH as DATASET_PATH

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TRAIN_PATH = Path("datasets/clean/topic_train.csv")
TEST_PATH = Path("datasets/clean/topic_test.csv")
TEXT_COL = "Text"

RANDOM_SEED = 42
TEST_SIZE = 0.2
MIN_WORDS = 15

# shortcut: stopword-ratio language check, swap for langdetect if it misfires
EN_STOPWORDS = {"the", "and", "of", "to", "a", "in", "is", "her", "his", "he", "she", "with", "for"}
MIN_STOPWORD_RATIO = 0.08

LEAK_RE = re.compile(r"\bbann?ed\b|\bban\b|\bchallenged\b|censor|\baward|\bprize\b|bestsell|best[- ]sell"
                     r"|new york times|starred review|\bpraise", re.I)
# "Quoted praise." —Source Name
REVIEW_RE = re.compile(r"[\"“][^\"”]{1,600}[\"”]\s*[-—–]+\s*[^.!?\"“]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def is_english(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and sum(w in EN_STOPWORDS for w in words) / len(words) >= MIN_STOPWORD_RATIO


def drop_leaky_sentences(text: str) -> str:
    text = REVIEW_RE.sub(" ", text)
    return " ".join(s for s in SENTENCE_RE.split(text) if not LEAK_RE.search(s))


def remove_title(text: str, title: str) -> str:
    # also the main title alone, since summaries rarely repeat the subtitle
    for t in {title, title.split(":")[0]}:
        t = t.strip()
        if len(t) >= 4:
            text = re.sub(re.escape(t), " ", text, flags=re.I)
    return text


def remove_persons(doc) -> str:
    text = doc.text
    for ent in reversed(doc.ents):
        if ent.label_ == "PERSON":
            text = text[:ent.start_char] + text[ent.end_char:]
    return text


def scrub(texts: list[str], titles: list[str], nlp) -> list[str]:
    texts = [remove_title(drop_leaky_sentences(_ascii_fold(t)), _ascii_fold(ti))
             for t, ti in zip(texts, titles)]
    texts = [remove_persons(doc) for doc in nlp.pipe(texts, batch_size=64)]
    return [re.sub(r"\s+", " ", re.sub(r"\s+([,.;:!?])", r"\1", t)).strip() for t in texts]


def _self_check(nlp) -> None:
    out = scrub(['Harry Potter finds out he is a wizard. "A delight!" —Kirkus Reviews '
                 "The novel was banned in 2022. Magic school awaits."], ["Harry Potter"], nlp)[0]
    assert "Harry" not in out and "banned" not in out and "Kirkus" not in out, out
    assert "wizard" in out and "Magic school awaits." in out, out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def build_corpus(df: pd.DataFrame, nlp, min_words: int = MIN_WORDS) -> pd.DataFrame:
    df = df.copy()
    df[BANNED_COL] = df[BANNED_COL] == "True"
    df[TEXT_COL] = df["Goodreads Summary"].fillna(df["CMU Summary"])

    def report(mask: pd.Series, reason: str) -> None:
        print(f"Dropped {reason:<12}: {mask.sum():>5}  "
              f"(banned {(mask & df[BANNED_COL]).sum()}, control {(mask & ~df[BANNED_COL]).sum()})")

    non_en = ~df[TEXT_COL].map(is_english)
    report(non_en, "non-English")
    df = df[~non_en].copy()

    df[TEXT_COL] = scrub(df[TEXT_COL].tolist(), df["Title"].fillna("").tolist(), nlp)

    short = df[TEXT_COL].str.split().str.len() < min_words
    report(short, "too short")
    df = df[~short]

    dup = df.duplicated(TEXT_COL)
    report(dup, "duplicate")
    return df[~dup][[ID_COL, BANNED_COL, TEXT_COL]].reset_index(drop=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Clean summaries for topic modelling and split train/test.")
    ap.add_argument("--input", type=Path, default=DATASET_PATH)
    ap.add_argument("--train-out", type=Path, default=TRAIN_PATH)
    ap.add_argument("--test-out", type=Path, default=TEST_PATH)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--test-size", type=float, default=TEST_SIZE)
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = ap.parse_args()

    # only NER is needed
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    _self_check(nlp)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, na_values=[""])
    print(f"Loaded     : {len(df)} books")
    corpus = build_corpus(df, nlp, args.min_words)

    train, test = train_test_split(corpus, test_size=args.test_size,
                                   stratify=corpus[BANNED_COL], random_state=args.seed)
    for name, part, path in [("train", train, args.train_out), ("test", test, args.test_out)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        part.sort_values(ID_COL, key=lambda s: s.astype(int)).to_csv(path, index=False)
        words = part[TEXT_COL].str.split().str.len().groupby(part[BANNED_COL]).median()
        print(f"{name:<5}: {len(part)} books (banned {part[BANNED_COL].sum()}), "
              f"median words banned {words[True]:.0f} / control {words[False]:.0f} -> {path}")


if __name__ == "__main__":
    main()
