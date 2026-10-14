"""
Train a kNN and a Logistic Regression model on the BERTopic topic distributions
(share of each topic per book) to predict banned vs not banned, and evaluate
both against a majority-class baseline on the held-out test set.
"""

import argparse

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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
#TODO: Use argparse for the file paths: dataset, topic vectors, train/test split,
#      topic names, output folder.
#TODO: Set the other settings as constants: random seed (42), number of CV folds (5),
#      grid of k values, grid of C values, classification threshold (0.5).

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
#TODO: Define a named function that turns cosine distances into similarities
#      (1 - distance), used as the kNN weights. It must be a named function,
#      not a lambda, or the model cannot be saved with joblib.

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
#TODO: Load the dataset (Book ID + Banned Status) and the topic vectors.
#TODO: Merge them on Book ID (never on row position).
#TODO: Load the topic names from the BERTopic output (needed for the weight table).
#TODO: Load the train/test split used for BERTopic and divide the data into
#      X_train, y_train, X_test, y_test (keep the Book IDs alongside).

# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
#TODO: Find books whose topic vector is all zeros, in BOTH the training and test set.
#TODO: Remove them and print how many were removed per set and per group
#      (banned / not banned), for the report.

# --------------------------------------------------------------------------- #
# Cross-validation folds
# --------------------------------------------------------------------------- #
#TODO: Create one StratifiedKFold (5 folds, shuffle=True, fixed seed) to use for
#      both models, so they are tuned the same way.

# --------------------------------------------------------------------------- #
# kNN model
# --------------------------------------------------------------------------- #
#TODO: Set up KNeighborsClassifier with metric="cosine", algorithm="brute" and the
#      similarity weight function.
#TODO: Tune k with GridSearchCV (grid of k values, the shared folds,
#      scoring="roc_auc"), fitted on the training set only.
#TODO: Print best_params_ and the best CV score; keep best_estimator_ as the kNN model.

# --------------------------------------------------------------------------- #
# Logistic Regression model
# --------------------------------------------------------------------------- #
#TODO: Set up LogisticRegression with max_iter=1000.
#TODO: Tune C with GridSearchCV (grid of C values, the shared folds,
#      scoring="roc_auc"), fitted on the training set only.
#TODO: Print best_params_ and the best CV score; keep best_estimator_.
#TODO: Take coef_[0] and intercept_ from the best estimator, put the weights in a
#      DataFrame next to the topic names, sort them, and print the top and bottom topics.

# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
#TODO: Fit DummyClassifier(strategy="most_frequent") on the training set.

# --------------------------------------------------------------------------- #
# Evaluation (test set, used only from here on)
# --------------------------------------------------------------------------- #
#TODO: For kNN, LogReg and the baseline: get scores with predict_proba(X_test)[:, 1]
#      and labels with predict(X_test).
#TODO: Compute precision, recall, macro-F1, micro-F1 (= accuracy in binary) and
#      ROC-AUC (from the scores, not the labels) and collect them in one table.
#TODO: Print classification_report and confusion_matrix per model.
#TODO: Plot the ROC curves of all models in one figure (RocCurveDisplay.from_predictions).

# --------------------------------------------------------------------------- #
# Explanations for example books
# --------------------------------------------------------------------------- #
#TODO: Pick a few test books (e.g. correct and incorrect predictions).
#TODO: kNN: use kneighbors() to list their nearest neighbours (titles, labels, similarity).
#TODO: LogReg: multiply the book's topic vector by coef_[0] and show the topics
#      with the largest contributions.

# --------------------------------------------------------------------------- #
# Save results
# --------------------------------------------------------------------------- #
#TODO: Save both models with joblib.
#TODO: Save the best k and C, the sorted topic-weight table and the metrics table
#      as CSV files, and the ROC figure as an image.