"""
Train a kNN and a Logistic Regression model on the BERTopic topic distributions
(share of each topic per book) to predict banned vs not banned, and evaluate
both against a majority-class baseline on the held-out test set.
"""

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

# allow running as a file (python scripts/x.py) as well as a module (python -m scripts.x)
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge import BANNED_COL, ID_COL, OUTPUT_PATH as DATASET_PATH

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TRAIN_PATH = Path("datasets/clean/topic_train.csv")
TEST_PATH = Path("datasets/clean/topic_test.csv")
TOPICS_DIR = Path("datasets/topics")
TRAIN_TOPICS_PATH = TOPICS_DIR / "train_topic_dist.csv"
TEST_TOPICS_PATH = TOPICS_DIR / "test_topic_dist.csv"
TOPIC_INFO_PATH = TOPICS_DIR / "topic_info.csv"
OUT_DIR = Path("datasets/classification")

RANDOM_SEED = 42
CV_FOLDS = 5
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101, 151, 201]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train kNN and Logistic Regression on BERTopic topic shares.")
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--train-topics", type=Path, default=TRAIN_TOPICS_PATH)
    ap.add_argument("--test-topics", type=Path, default=TEST_TOPICS_PATH)
    ap.add_argument("--train", type=Path, default=TRAIN_PATH)
    ap.add_argument("--test", type=Path, default=TEST_PATH)
    ap.add_argument("--topic-info", type=Path, default=TOPIC_INFO_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return ap.parse_args()


args = parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def cosine_similarity_weights(distances: np.ndarray) -> np.ndarray:
    """kNN weights: cosine distance -> similarity (named, not a lambda, so joblib can pickle it)."""
    return 1.0 - distances


def sqrt_shares(X: np.ndarray) -> np.ndarray:
    """Square root of the topic shares: evens out the many tiny shares and the few large ones.
    Cosine on sqrt shares is the Bhattacharyya coefficient between two topic distributions."""
    return np.sqrt(X)


def sqrt_transform() -> FunctionTransformer:
    return FunctionTransformer(sqrt_shares, feature_names_out="one-to-one")


def best_score_where(search: GridSearchCV, **params) -> float:
    """Best mean CV score among the grid points matching the given parameter values."""
    res = search.cv_results_
    mask = np.ones(len(res["params"]), dtype=bool)
    for key, value in params.items():
        mask &= np.array([p[key] is value or p[key] == value for p in res["params"]])
    return float(res["mean_test_score"][mask].max())

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
dataset = pd.read_csv(args.dataset, usecols=[ID_COL, "Title", BANNED_COL])
dataset[BANNED_COL] = dataset[BANNED_COL].astype(bool)

# the label comes from the dataset, so drop the copy stored next to the topic shares
topics = pd.concat([pd.read_csv(args.train_topics).assign(Split="train"),
                    pd.read_csv(args.test_topics).assign(Split="test")], ignore_index=True)
topics = topics.drop(columns=BANNED_COL)
TOPIC_COLS = [c for c in topics.columns if c.startswith("topic_")]

data = dataset.merge(topics, on=ID_COL, how="inner", validate="one_to_one")
print(f"Merged {len(data)} books with {len(TOPIC_COLS)} topic shares")

topic_info = pd.read_csv(args.topic_info, usecols=["Topic", "Name"]).set_index("Topic")["Name"]
topic_names = [topic_info[int(c.removeprefix("topic_"))] for c in TOPIC_COLS]

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
# books with no topic at all have an all-zero vector: cosine distance is undefined for them
zero = (data[TOPIC_COLS] == 0).all(axis=1)
removed = (data[zero].groupby(["Split", BANNED_COL]).size()
           .unstack(fill_value=0).reindex(index=["train", "test"], columns=[True, False], fill_value=0)
           .rename(columns={True: "banned", False: "not banned"}))
removed["total"] = removed.sum(axis=1)
print(f"Removed {int(zero.sum())} books with an all-zero topic vector:")
print(removed.to_string())

data = data[~zero].reset_index(drop=True)

train = data[data["Split"] == "train"].reset_index(drop=True)
test = data[data["Split"] == "test"].reset_index(drop=True)
X_train, y_train = train[TOPIC_COLS].to_numpy(), train[BANNED_COL].to_numpy()
X_test, y_test = test[TOPIC_COLS].to_numpy(), test[BANNED_COL].to_numpy()
print(f"Train: {len(train)} books, test: {len(test)} books")

# --------------------------------------------------------------------------- #
# Cross-validation folds
# --------------------------------------------------------------------------- #
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

# --------------------------------------------------------------------------- #
# kNN model
# --------------------------------------------------------------------------- #
knn_search = GridSearchCV(
    Pipeline([("transform", "passthrough"),
              ("knn", KNeighborsClassifier(metric="cosine", algorithm="brute"))]),
    param_grid={
        "transform": ["passthrough", sqrt_transform()],
        "knn__n_neighbors": K_GRID,
        "knn__weights": ["uniform", cosine_similarity_weights],
    },
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
knn_search.fit(X_train, y_train)
print(f"\nkNN best params: {knn_search.best_params_}, CV ROC-AUC: {knn_search.best_score_:.3f}")
print(f"  previous setup (raw shares, similarity weights): "
      f"{best_score_where(knn_search, transform='passthrough', knn__weights=cosine_similarity_weights):.3f}")
knn = knn_search.best_estimator_

# --------------------------------------------------------------------------- #
# Logistic Regression model
# --------------------------------------------------------------------------- #
# pairs: one feature per topic pair (product of the two shares), so weights stay readable as topics
logreg_search = GridSearchCV(
    Pipeline([("transform", "passthrough"),
              ("pairs", "passthrough"),
              ("scale", StandardScaler()),
              ("logreg", LogisticRegression(solver="saga", max_iter=5000))]),
    param_grid={
        "transform": ["passthrough", sqrt_transform()],
        "pairs": ["passthrough", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)],
        "logreg__l1_ratio": [1.0, 0.0],  # 1 = L1 (sparse), 0 = L2
        "logreg__C": C_GRID,
    },
    cv=cv, scoring="roc_auc", n_jobs=-1,
)
logreg_search.fit(X_train, y_train)
print(f"LogReg best params: {logreg_search.best_params_}, CV ROC-AUC: {logreg_search.best_score_:.3f}")
print(f"  previous setup (raw shares, no pairs, L2): "
      f"{best_score_where(logreg_search, transform='passthrough', pairs='passthrough', logreg__l1_ratio=0.0):.3f}")
logreg = logreg_search.best_estimator_

# features after transform/pairs: a topic name, or two topic names for a pair
feature_names = [n.replace(" ", " x ") for n in logreg[:-1].get_feature_names_out(topic_names)]
coef, intercept = logreg[-1].coef_[0], logreg[-1].intercept_[0]
weights = (pd.DataFrame({"feature": feature_names, "weight": coef})
           .sort_values("weight", ascending=False).reset_index(drop=True))
print(f"\nLogReg intercept: {intercept:.3f}, non-zero weights: {int((coef != 0).sum())} of {len(coef)}")
print("(features are standardised: a weight is the effect of a one-SD increase in that feature)")
print("Features pushing towards banned:")
print(weights.head(10).to_string(index=False))
print("Features pushing towards not banned:")
print(weights.tail(10).iloc[::-1].to_string(index=False))

# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

# --------------------------------------------------------------------------- #
# Evaluation (test set, used only from here on)
# --------------------------------------------------------------------------- #
models = {"kNN": knn, "LogReg": logreg, "Baseline": baseline}
scores = {name: m.predict_proba(X_test)[:, 1] for name, m in models.items()}
labels = {name: m.predict(X_test) for name, m in models.items()}

metrics = pd.DataFrame([
    {
        "model": name,
        "precision": precision_score(y_test, labels[name], zero_division=0),
        "recall": recall_score(y_test, labels[name]),
        "macro_f1": f1_score(y_test, labels[name], average="macro"),
        "micro_f1": f1_score(y_test, labels[name], average="micro"),
        "roc_auc": roc_auc_score(y_test, scores[name]),
    }
    for name in models
])
print("\nTest metrics:")
print(metrics.round(3).to_string(index=False))

for name in models:
    print(f"\n--- {name} ---")
    print(classification_report(y_test, labels[name], target_names=["not banned", "banned"], zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted; not banned, banned):")
    print(confusion_matrix(y_test, labels[name]))

fig, ax = plt.subplots(figsize=(6, 6))
for i, name in enumerate(models):
    RocCurveDisplay.from_predictions(y_test, scores[name], name=name, ax=ax,
                                     plot_chance_level=(i == len(models) - 1))
ax.set_title("ROC curves on the test set")
fig.tight_layout()

# --------------------------------------------------------------------------- #
# Explanations for example books
# --------------------------------------------------------------------------- #
# one book per outcome of the LogReg prediction: TP, TN, FP, FN
pred = labels["LogReg"]
outcomes = {
    "correct banned": (y_test & pred),
    "correct not banned": (~y_test & ~pred),
    "wrongly predicted banned": (~y_test & pred),
    "wrongly predicted not banned": (y_test & ~pred),
}
examples = {desc: np.flatnonzero(mask)[0] for desc, mask in outcomes.items() if mask.any()}

for desc, i in examples.items():
    book = test.iloc[i]
    print(f"\n=== {book['Title']} (Book ID {book[ID_COL]}) - {desc} ===")
    print(f"True: {'banned' if y_test[i] else 'not banned'}, "
          f"kNN score: {scores['kNN'][i]:.3f}, LogReg score: {scores['LogReg'][i]:.3f}")

    dist, idx = knn[-1].kneighbors(knn[:-1].transform(X_test[i:i + 1]), n_neighbors=10)
    neighbours = pd.DataFrame({
        "title": train["Title"].iloc[idx[0]].to_numpy(),
        "banned": y_train[idx[0]],
        "similarity": cosine_similarity_weights(dist[0]).round(3),
    })
    print(f"kNN 10 nearest neighbours (of the {knn[-1].n_neighbors} used):")
    print(neighbours.to_string(index=False))

    x = logreg[:-1].transform(X_test[i:i + 1])[0]
    contrib = (pd.DataFrame({"feature": feature_names, "value": x, "contribution": x * coef})
               .sort_values("contribution", key=np.abs, ascending=False).head(5))
    print(f"LogReg largest feature contributions (intercept {intercept:.3f}):")
    print(contrib.round(3).to_string(index=False))

# --------------------------------------------------------------------------- #
# Save results
# --------------------------------------------------------------------------- #
joblib.dump(knn, args.out_dir / "knn.joblib")
joblib.dump(logreg, args.out_dir / "logreg.joblib")

pd.DataFrame([{"model": name, "param": key, "value": str(value), "cv_roc_auc": search.best_score_}
              for name, search in {"kNN": knn_search, "LogReg": logreg_search}.items()
              for key, value in search.best_params_.items()]) \
    .to_csv(args.out_dir / "best_params.csv", index=False)
for name, search in {"knn": knn_search, "logreg": logreg_search}.items():
    res = pd.DataFrame(search.cv_results_)
    res["params"] = res["params"].astype(str)
    res.filter(regex="^(param|mean_test|std_test|rank_test)").astype(str) \
        .to_csv(args.out_dir / f"{name}_cv_results.csv", index=False)
weights.to_csv(args.out_dir / "topic_weights.csv", index=False)
metrics.to_csv(args.out_dir / "metrics.csv", index=False)
fig.savefig(args.out_dir / "roc_curves.png", dpi=150)
print(f"\nSaved models, tables and ROC figure to {args.out_dir}")