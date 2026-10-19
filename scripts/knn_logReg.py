"""
Train a kNN and a Logistic Regression model on the BERTopic topic distributions
(share of each topic per book) to predict banned vs not banned, and evaluate
both against a majority-class baseline on the held-out test set.
"""

import argparse
import sys
import time
from contextlib import redirect_stdout
from datetime import timedelta
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
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from tqdm import tqdm

# allow running as a file (python scripts/x.py) as well as a module (python -m scripts.x)
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.merge import BANNED_COL, ID_COL, OUTPUT_PATH as DATASET_PATH

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TOPICS_ROOT = Path("datasets/topics/topics_m")
RUNS = ["14t", "17t", "19t", "24t", "28t", "45t", "72t", "88t", "104t", "126t", "149t"]  # one BERTopic model per sub-folder of TOPICS_ROOT
OUT_ROOT = Path("datasets/topics")

RANDOM_SEED = 42
CV_FOLDS = 5
K_GRID = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101, 151, 201, 301, 501]
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1, 10]
# topic pairs grow quadratically (72 topics -> 2556 pairs) and make LogReg take hours; they only
# helped up to 28 topics and lowered the CV ROC-AUC from 45 topics on, so they are skipped above this
PAIRS_MAX_TOPICS = 30
THRESHOLD = 0.5

# figure palette
CRIMSON, NAVY, CREAM, SLATE = "#CC2255", "#1E2433", "#E8E1D3", "#23395B"
ROC_STYLE = {
    "kNN": {"color": CRIMSON, "linestyle": "-"},
    "LogReg": {"color": SLATE, "linestyle": "-"},
    "Baseline": {"color": NAVY, "linestyle": "--"},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train kNN and Logistic Regression on BERTopic topic shares.")
    ap.add_argument("--dataset", type=Path, default=DATASET_PATH)
    ap.add_argument("--topics-root", type=Path, default=TOPICS_ROOT)
    ap.add_argument("--runs", nargs="+", default=RUNS, help="sub-folders of --topics-root, e.g. 6t 11t")
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--plot-only", action="store_true",
                    help="only redraw the parameter search figure from the saved CV results of each run")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
class Tee:
    """Write to several streams at once, e.g. the terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def fit_with_progress(search: GridSearchCV, X: np.ndarray, y: np.ndarray, desc: str) -> GridSearchCV:
    """Fit a grid search with a progress bar (one tick per CV fit), shown on stderr so it stays out of the logs.
    joblib reports finished fits through Parallel.print_progress, so it is swapped for a bar update while fitting."""
    n_fits = len(ParameterGrid(search.param_grid)) * search.cv.get_n_splits()
    original = joblib.Parallel.print_progress
    with tqdm(total=n_fits, desc=desc, unit="fit", file=sys.stderr) as bar:
        joblib.Parallel.print_progress = lambda parallel: bar.update(parallel.n_completed_tasks - bar.n)
        try:
            search.fit(X, y)
        finally:
            joblib.Parallel.print_progress = original
    return search


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


def style_axes(ax: plt.Axes) -> None:
    """Figure palette on one axes: cream background, navy text, ticks and frame."""
    ax.set_facecolor(CREAM)
    ax.title.set_color(NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.tick_params(colors=NAVY)
    for spine in ax.spines.values():
        spine.set_color(NAVY)


def plot_param_search(out_dir: Path) -> None:
    """Mean CV ROC-AUC of every grid point (from the saved *_cv_results.csv) with the chosen one marked.
    Colour = transform (raw / sqrt shares), line style = kNN weights or LogReg penalty, marker = LogReg pairs."""
    # the saved parameter values are reprs, e.g. "FunctionTransformer(... sqrt_shares ...)": map them to short labels
    def short(col: str, value: str) -> str:
        if "sqrt_shares" in value:
            return "sqrt shares"
        if "cosine_similarity_weights" in value:
            return "similarity"
        if "PolynomialFeatures" in value:
            return "pairs"
        if value == "passthrough":
            return "no pairs" if col == "param_pairs" else "raw shares"
        return {"1.0": "L1", "0.0": "L2"}.get(value, value)

    panels = {  # model: (results file, x-axis parameter, x label, parameter for the line style)
        "kNN": ("knn_cv_results.csv", "param_knn__n_neighbors", "k (neighbours)", "param_knn__weights"),
        "LogReg": ("logreg_cv_results.csv", "param_logreg__C", "C (inverse regularisation)", "param_logreg__l1_ratio"),
    }
    colors = {"raw shares": SLATE, "sqrt shares": CRIMSON}
    linestyles = ["-", "--"]
    n_topics = out_dir.name.removeprefix("classification_").removesuffix("topics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=CREAM)
    for ax, (model, (file, x_col, x_label, style_col)) in zip(axes, panels.items()):
        res = pd.read_csv(out_dir / file, dtype=str)
        res[x_col] = res[x_col].astype(float)
        res[["mean_test_score", "std_test_score"]] = res[["mean_test_score", "std_test_score"]].astype(float)
        res["rank_test_score"] = res["rank_test_score"].astype(int)
        group_cols = [c for c in res.columns if c.startswith("param_") and c != x_col]
        for c in group_cols:
            res[c] = res[c].map(lambda v: short(c, v))
        style_values = list(res[style_col].unique())
        # a parameter with a single tested value (pairs above PAIRS_MAX_TOPICS) is left out of the labels
        label_cols = [c for c in group_cols if res[c].nunique() > 1]

        for key, grp in res.groupby(group_cols, sort=False):
            setting = dict(zip(group_cols, key))
            grp = grp.sort_values(x_col)
            color = colors[setting["param_transform"]]
            ax.fill_between(grp[x_col], grp["mean_test_score"] - grp["std_test_score"],
                            grp["mean_test_score"] + grp["std_test_score"], color=color, alpha=0.08, linewidth=0)
            ax.plot(grp[x_col], grp["mean_test_score"], color=color,
                    linestyle=linestyles[style_values.index(setting[style_col])],
                    marker="s" if setting.get("param_pairs") == "pairs" else "o", markersize=4, linewidth=1.8,
                    label=", ".join(setting[c] for c in label_cols))

        best = res.loc[res["rank_test_score"].idxmin()]
        chosen = [f"{x_label.split()[0]} = {best[x_col]:g}"] + [best[c] for c in label_cols]
        ax.scatter(best[x_col], best["mean_test_score"], marker="*", s=320, color=NAVY, edgecolor=CREAM, zorder=5,
                   label=f"chosen: {', '.join(chosen)}\n(CV ROC-AUC {best['mean_test_score']:.3f})")

        ax.set_xscale("log")
        ax.set_xticks(sorted(res[x_col].unique()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(res[x_col].unique())], rotation=45)
        ax.minorticks_off()
        ax.set_xlabel(x_label)
        ax.set_ylabel("mean CV ROC-AUC (band = ± 1 SD over folds)")
        ax.set_title(f"{model} grid search", color=NAVY)
        ax.grid(True, color=NAVY, alpha=0.1)
        style_axes(ax)
        ax.legend(loc="lower right", fontsize=8, facecolor=CREAM, edgecolor=NAVY, labelcolor=NAVY)

    fig.suptitle(f"Parameters tested and chosen ({n_topics} topics, {CV_FOLDS}-fold CV)", color=NAVY)
    fig.tight_layout()
    fig.savefig(out_dir / "param_search.png", dpi=150)
    plt.close(fig)


def plot_auc_by_topics(out_root: Path) -> None:
    """Test ROC-AUC (from classification_metrics_by_topics.csv) and best CV ROC-AUC (from each run's
    best_params.csv) of kNN and LogReg against the number of topics."""
    test = pd.read_csv(out_root / "classification_metrics_by_topics.csv")
    cv = pd.concat([pd.read_csv(out_root / f"classification_{n}topics" / "best_params.csv")
                    .drop_duplicates("model").assign(topics=n) for n in test["topics"].unique()])
    topics = sorted(test["topics"].unique())

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=CREAM)
    for model in ["kNN", "LogReg"]:
        color = ROC_STYLE[model]["color"]
        t = test[test["model"] == model].sort_values("topics")
        c = cv[cv["model"] == model].sort_values("topics")
        ax.plot(t["topics"], t["roc_auc"], color=color, marker="o", linewidth=2, label=f"{model} (test)")
        ax.plot(c["topics"], c["cv_roc_auc"], color=color, marker="o", markerfacecolor=CREAM, linestyle="--",
                linewidth=1.5, label=f"{model} ({CV_FOLDS}-fold CV, best params)")
        best = t.loc[t["roc_auc"].idxmax()]
        ax.annotate(f"{best['roc_auc']:.3f}", xy=(best["topics"], best["roc_auc"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", color=color, fontsize=9)
    ax.axhline(0.5, color=NAVY, linestyle=":", linewidth=1, alpha=0.6, label="Baseline (majority class)")

    ax.set_xscale("log")
    ax.set_xticks(topics)
    ax.set_xticklabels([str(n) for n in topics])
    ax.minorticks_off()
    ax.set_xlabel("number of BERTopic topics")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("ROC-AUC of kNN and LogReg by number of topics")
    ax.grid(True, color=NAVY, alpha=0.1)
    style_axes(ax)
    ax.legend(loc="lower right", fontsize=9, facecolor=CREAM, edgecolor=NAVY, labelcolor=NAVY)
    fig.tight_layout()
    fig.savefig(out_root / "roc_auc_by_topics.png", dpi=150)
    plt.close(fig)


def run(topics_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Train and evaluate kNN and LogReg on one BERTopic model's topic shares; returns the test metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------- #
    # Loading
    # --------------------------------------------------------------------------- #
    dataset = pd.read_csv(args.dataset, usecols=[ID_COL, "Title", BANNED_COL])
    dataset[BANNED_COL] = dataset[BANNED_COL].astype(bool)

    # the label comes from the dataset, so drop the copy stored next to the topic shares
    topics = pd.concat([pd.read_csv(topics_dir / "train_topic_dist.csv").assign(Split="train"),
                        pd.read_csv(topics_dir / "test_topic_dist.csv").assign(Split="test")], ignore_index=True)
    topics = topics.drop(columns=BANNED_COL)
    TOPIC_COLS = [c for c in topics.columns if c.startswith("topic_")]

    data = dataset.merge(topics, on=ID_COL, how="inner", validate="one_to_one")
    print(f"Merged {len(data)} books with {len(TOPIC_COLS)} topic shares")

    topic_info = pd.read_csv(topics_dir / "topic_info.csv", usecols=["Topic", "Name"]).set_index("Topic")["Name"]
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
    fit_with_progress(knn_search, X_train, y_train, "kNN grid search")
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
            "pairs": ["passthrough"] + ([PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)]
                                        if len(TOPIC_COLS) <= PAIRS_MAX_TOPICS else []),
            "logreg__l1_ratio": [1.0, 0.0],  # 1 = L1 (sparse), 0 = L2
            "logreg__C": C_GRID,
        },
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    fit_with_progress(logreg_search, X_train, y_train, "LogReg grid search")
    print(f"LogReg best params: {logreg_search.best_params_}, CV ROC-AUC: {logreg_search.best_score_:.3f}")
    print(f"  previous setup (raw shares, no pairs, L2): "
          f"{best_score_where(logreg_search, transform='passthrough', pairs='passthrough', logreg__l1_ratio=0.0):.3f}")
    logreg = logreg_search.best_estimator_
    n_iter, max_iter = int(logreg[-1].n_iter_[0]), logreg[-1].max_iter
    print(f"  best LogReg {'did NOT converge' if n_iter >= max_iter else 'converged'} "
          f"({n_iter} of {max_iter} iterations)")

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

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    for i, name in enumerate(models):
        RocCurveDisplay.from_predictions(y_test, scores[name], name=name, ax=ax,
                                         curve_kwargs={**ROC_STYLE[name], "linewidth": 2},
                                         plot_chance_level=(i == len(models) - 1),
                                         chance_level_kw={"color": NAVY, "linestyle": ":", "linewidth": 1, "alpha": 0.6})
    ax.set_title(f"ROC curves on the test set ({len(TOPIC_COLS)} topics)", color=NAVY)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.tick_params(colors=NAVY)
    for spine in ax.spines.values():
        spine.set_color(NAVY)
    ax.legend(loc="lower right", facecolor=CREAM, edgecolor=NAVY, labelcolor=NAVY)
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
    joblib.dump(knn, out_dir / "knn.joblib")
    joblib.dump(logreg, out_dir / "logreg.joblib")

    pd.DataFrame([{"model": name, "param": key, "value": str(value), "cv_roc_auc": search.best_score_}
                  for name, search in {"kNN": knn_search, "LogReg": logreg_search}.items()
                  for key, value in search.best_params_.items()]) \
        .to_csv(out_dir / "best_params.csv", index=False)
    for name, search in {"knn": knn_search, "logreg": logreg_search}.items():
        res = pd.DataFrame(search.cv_results_)
        res["params"] = res["params"].astype(str)
        res.filter(regex="^(param|mean_test|std_test|rank_test)").astype(str) \
            .to_csv(out_dir / f"{name}_cv_results.csv", index=False)
    weights.to_csv(out_dir / "topic_weights.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)
    plot_param_search(out_dir)
    print(f"\nSaved models, tables, ROC and parameter search figures to {out_dir}")
    return metrics


args = parse_args()
args.out_root.mkdir(parents=True, exist_ok=True)

if args.plot_only:
    for run_name in args.runs:
        out_dir = args.out_root / f"classification_{run_name.removesuffix('t')}topics"
        plot_param_search(out_dir)
        print(f"Saved {out_dir / 'param_search.png'}")
    plot_auc_by_topics(args.out_root)
    print(f"Saved {args.out_root / 'roc_auc_by_topics.png'}")
    sys.exit()

# everything printed goes to the terminal and to a log file: one for the whole script, one per run
with open(args.out_root / "classification_log.txt", "w", encoding="utf-8") as log, \
        redirect_stdout(Tee(sys.stdout, log)):
    all_metrics, run_times = [], {}
    script_start = time.perf_counter()
    for i, run_name in enumerate(args.runs, start=1):
        n_topics = run_name.removesuffix("t")
        out_dir = args.out_root / f"classification_{n_topics}topics"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "log.txt", "w", encoding="utf-8") as run_log, \
                redirect_stdout(Tee(sys.stdout, run_log)):
            print(f"\n{'#' * 79}\n# [{i}/{len(args.runs)}] {run_name}: {args.topics_root / run_name}\n{'#' * 79}")
            run_start = time.perf_counter()
            metrics = run(args.topics_root / run_name, out_dir)
            run_times[run_name] = time.perf_counter() - run_start
            print(f"\n{run_name} took {format_duration(run_times[run_name])} "
                  f"(total so far {format_duration(time.perf_counter() - script_start)})")
        all_metrics.append(metrics.assign(topics=int(n_topics)))

    summary = pd.concat(all_metrics, ignore_index=True).set_index(["topics", "model"])
    print("\nTest metrics per number of topics:")
    print(summary.round(3).to_string())
    print("\nTime per run:")
    for run_name, seconds in run_times.items():
        print(f"  {run_name:>4}: {format_duration(seconds)}")
    print(f"  total: {format_duration(time.perf_counter() - script_start)}")
    summary.to_csv(args.out_root / "classification_metrics_by_topics.csv")
    plot_auc_by_topics(args.out_root)
