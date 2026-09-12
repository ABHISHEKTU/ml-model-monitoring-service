"""
Multivariate drift via domain classifier (classifier two-sample test).

CONCEPT:
- Label reference rows 0, current rows 1. Train a classifier to tell them apart
  using ALL features jointly (not one at a time like KS/PSI).
- If reference and current are truly the same distribution, the classifier can't
  do better than random guessing -> AUC ~ 0.5.
- If AUC is meaningfully above 0.5, the classifier found some joint pattern that
  separates old data from new data -> multivariate drift, even when every
  individual feature's marginal distribution looks unchanged.
- Why this catches what KS/PSI miss: correlation/relationship changes between
  features (e.g. age and income used to move together, now don't) don't show up
  in any single-feature test, but a classifier trained on both columns together
  can pick up on the changed relationship.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from app.exceptions import InsufficientDataError

MIN_ROWS_PER_SIDE = 30
AUC_DRIFT_THRESHOLD = 0.65  # meaningfully above 0.5 = classifier found a real pattern


def multivariate_drift_test(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                             numeric_features: list[str], categorical_features: list[str]) -> dict:
    if len(reference_df) < MIN_ROWS_PER_SIDE or len(current_df) < MIN_ROWS_PER_SIDE:
        raise InsufficientDataError(
            "multivariate", min(len(reference_df), len(current_df)), MIN_ROWS_PER_SIDE
        )

    ref = reference_df.copy()
    cur = current_df.copy()
    ref["__label__"] = 0
    cur["__label__"] = 1
    combined = pd.concat([ref, cur], ignore_index=True)

    # Build feature matrix: numeric as-is, categorical one-hot encoded.
    X_numeric = combined[numeric_features].fillna(combined[numeric_features].median())
    X_categorical = pd.get_dummies(combined[categorical_features], dummy_na=True) if categorical_features else pd.DataFrame(index=combined.index)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = combined["__label__"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    return {
        "test": "domain_classifier",
        "auc": round(float(auc), 6),
        "drift_detected": bool(auc >= AUC_DRIFT_THRESHOLD),
        "threshold": AUC_DRIFT_THRESHOLD,
        "interpretation": (
            "Classifier could reliably distinguish reference from current data "
            "using the joint feature relationships — multivariate drift detected."
            if auc >= AUC_DRIFT_THRESHOLD else
            "Classifier could not reliably distinguish reference from current data "
            "(AUC near 0.5) — no multivariate drift detected."
        ),
        "n_reference": int(len(reference_df)),
        "n_current": int(len(current_df)),
    }