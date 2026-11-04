"""
Linear and Trivial Baselines for Multi-Label Framing
CS 5134/6034 NLP Fall 2026 - Team FrameVision
Authors: Leo Kao, En-Shuo Zhang
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit_predict_ovr(X_train, Y_train, X_test, C=1.0):
    """
    One logistic regression per label, returning test probabilities. A label that is
    always absent (or always present) in training gets a constant probability.
    """
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    probs = np.zeros((len(X_test), Y_train.shape[1]))
    for c in range(Y_train.shape[1]):
        if Y_train[:, c].min() == Y_train[:, c].max():
            probs[:, c] = float(Y_train[0, c])
            continue
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
        probs[:, c] = clf.fit(X_train, Y_train[:, c]).predict_proba(X_test)[:, 1]
    return probs


def tfidf_fit_predict(train_texts, Y_train, eval_text_lists, C=10.0):
    """Fits TF-IDF on training text only, then returns probabilities per evaluation set."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    X_train = vectorizer.fit_transform(train_texts)
    results = []
    for texts in eval_text_lists:
        X_eval = vectorizer.transform(texts)
        probs = np.zeros((X_eval.shape[0], Y_train.shape[1]))
        for c in range(Y_train.shape[1]):
            if Y_train[:, c].min() == Y_train[:, c].max():
                probs[:, c] = float(Y_train[0, c])
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
            probs[:, c] = clf.fit(X_train, Y_train[:, c]).predict_proba(X_eval)[:, 1]
        results.append(probs)
    return results


def most_frequent_label_predict(Y_train, n_test):
    """Predicts the single most frequent training frame for every example."""
    pred = np.zeros((n_test, Y_train.shape[1]), dtype=int)
    pred[:, int(np.argmax(Y_train.sum(axis=0)))] = 1
    return pred


def sample_label_prior(Y_train, n_test, rng):
    """Random baseline: samples each frame from its training frequency."""
    return (rng.rand(n_test, Y_train.shape[1]) < Y_train.mean(axis=0)).astype(int)
