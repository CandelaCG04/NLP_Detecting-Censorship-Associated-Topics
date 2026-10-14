#!/usr/bin/env python3
"""
Fit BERTopic on the training summaries (no ban labels) and represent every book
as a vector of topic shares.

Outputs, in --out-dir:

    model/                  fitted BERTopic model (safetensors)
    topic_info.csv          topic id, size, top words, representative docs
    train_topic_dist.csv    Book ID, Banned Status, topic_0 .. topic_K-1 (rows sum to 1)
    test_topic_dist.csv     same, for the test books (transform only, never fitted)
    stability.csv           ARI / top-word Jaccard between refits with other seeds
    embeddings_*.npy        cached sentence embeddings

"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

# allow running as a file (python scripts/x.py) as well as a module (python -m scripts.x)
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge import BANNED_COL, ID_COL
from scripts.text_preprocess import TEST_PATH, TEXT_COL, TRAIN_PATH

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OUT_DIR = Path("datasets/topics")
EMBEDDING_MODEL = "all-mpnet-base-v2"
MIN_TOPIC_SIZE = 25
RANDOM_SEED = 42
TOP_WORDS = 10


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def embed(docs: list[str], encoder: SentenceTransformer, cache: Path) -> np.ndarray:
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(docs):
            return emb
    emb = encoder.encode(docs, batch_size=32, show_progress_bar=True)
    np.save(cache, emb)
    return emb


def _vectorizer() -> CountVectorizer:
    return CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=5)


def fit_model(docs: list[str], emb: np.ndarray, encoder, seed: int, min_topic_size: int) -> BERTopic:
    model = BERTopic(
        embedding_model=encoder,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=_vectorizer(),
    )
    topics, _ = model.fit_transform(docs, emb)
    # without this, topic -1 holds a large share of the books
    if -1 in topics:
        topics = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=emb)
        model.update_topics(docs, topics=topics, vectorizer_model=_vectorizer())
    return model


def topic_shares(model: BERTopic, docs: list[str], assigned: list[int]) -> np.ndarray:
    dist, _ = model.approximate_distribution(docs, window=4, stride=1)
    dist = np.asarray(dist, dtype=float)
    # docs where no token window matched any topic: fall back to the hard assignment
    for i in np.where(dist.sum(axis=1) == 0)[0]:
        dist[i, assigned[i]] = 1.0
    return dist / dist.sum(axis=1, keepdims=True)


def save_dist(df: pd.DataFrame, dist: np.ndarray, path: Path) -> None:
    cols = pd.DataFrame(dist, columns=[f"topic_{i}" for i in range(dist.shape[1])], index=df.index)
    pd.concat([df[[ID_COL, BANNED_COL]], cols], axis=1).to_csv(path, index=False, float_format="%.6f")


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def _top_words(model: BERTopic) -> list[set[str]]:
    return [{w for w, _ in model.get_topic(t)[:TOP_WORDS]} for t in model.get_topics() if t != -1]


def _best_jaccard(a: list[set[str]], b: list[set[str]]) -> float:
    return float(np.mean([max(len(x & y) / len(x | y) for y in b) for x in a]))


def stability(docs, emb, encoder, seeds: list[int], min_topic_size: int) -> pd.DataFrame:
    fits = {}
    for s in seeds:
        print(f"Stability refit, seed {s}")
        m = fit_model(docs, emb, encoder, s, min_topic_size)
        fits[s] = (m.topics_, _top_words(m))
    rows = []
    for a, b in itertools.combinations(seeds, 2):
        (ta, wa), (tb, wb) = fits[a], fits[b]
        rows.append({"seed_a": a, "seed_b": b, "topics_a": len(wa), "topics_b": len(wb),
                     "ari": adjusted_rand_score(ta, tb),
                     "word_jaccard": (_best_jaccard(wa, wb) + _best_jaccard(wb, wa)) / 2})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Fit BERTopic on training summaries and export topic shares.")
    ap.add_argument("--train", type=Path, default=TRAIN_PATH)
    ap.add_argument("--test", type=Path, default=TEST_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument("--min-topic-size", type=int, default=MIN_TOPIC_SIZE)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--stability-seeds", type=int, nargs="*", default=[],
                    help="extra UMAP seeds to refit with, e.g. 0 1 2 3 4")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    train_docs, test_docs = train[TEXT_COL].tolist(), test[TEXT_COL].tolist()

    encoder = SentenceTransformer(args.embedding_model)
    train_emb = embed(train_docs, encoder, args.out_dir / "embeddings_train.npy")
    test_emb = embed(test_docs, encoder, args.out_dir / "embeddings_test.npy")

    model = fit_model(train_docs, train_emb, encoder, args.seed, args.min_topic_size)
    info = model.get_topic_info()
    info.to_csv(args.out_dir / "topic_info.csv", index=False)
    print(f"Topics     : {(info['Topic'] != -1).sum()}")

    test_topics, _ = model.transform(test_docs, test_emb)
    # reduce_outliers refuses once the model has no -1 topic, so map test outliers to the closest topic directly
    nearest = cosine_similarity(test_emb, model.topic_embeddings_[model._outliers:]).argmax(axis=1)
    test_topics = np.where(np.asarray(test_topics) == -1, nearest, test_topics)
    save_dist(train, topic_shares(model, train_docs, model.topics_), args.out_dir / "train_topic_dist.csv")
    save_dist(test, topic_shares(model, test_docs, test_topics), args.out_dir / "test_topic_dist.csv")

    model.save(args.out_dir / "model", serialization="safetensors", save_ctfidf=True,
               save_embedding_model=args.embedding_model)

    if args.stability_seeds:
        stab = stability(train_docs, train_emb, encoder, args.stability_seeds, args.min_topic_size)
        stab.to_csv(args.out_dir / "stability.csv", index=False)
        print(stab.round(3).to_string(index=False))
        print(f"Mean ARI {stab['ari'].mean():.3f}, mean top-word Jaccard {stab['word_jaccard'].mean():.3f}")
    print(f"\nSaved -> {args.out_dir}")


if __name__ == "__main__":
    main()
