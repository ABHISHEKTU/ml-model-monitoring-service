"""
Population Stability Index (PSI).

CONCEPT:
- Bucket the reference distribution into bins (deciles by default for numeric —
  quantile-based, so each ref bin holds ~equal % of ref population).
- For each bin, compute % of reference pop in it, and % of current pop in it.
- PSI = sum over bins of (cur_pct - ref_pct) * ln(cur_pct / ref_pct)
- Thresholds (originally from credit scoring):
    PSI < 0.10            -> no significant shift
    0.10 <= PSI < 0.25     -> moderate shift, investigate
    PSI >= 0.25            -> major shift, retrain/investigate urgently
- PSI vs KS: PSI gives a MAGNITUDE, not a significance test. Less sensitive to
  sample size, and extends naturally to categorical features (bin = category).
- Zero-division guard: add a small epsilon to every bin % so ln(0) never happens.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.exceptions import InsufficientDataError

EPSILON = 1e-4
DEFAULT_BINS = 10


def _psi_from_percents(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.where(ref_pct == 0, EPSILON, ref_pct)
    cur_pct = np.where(cur_pct == 0, EPSILON, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_numeric(reference: np.ndarray, current: np.ndarray, feature_name: str,
                 bins: int = DEFAULT_BINS) -> dict:
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = np.asarray(current, dtype=float)
    cur = cur[~np.isnan(cur)]

    if len(ref) < bins * 2 or len(cur) < bins * 2:
        raise InsufficientDataError(feature_name, min(len(ref), len(cur)), bins * 2)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    psi_value = _psi_from_percents(ref_pct, cur_pct)
    return _build_result(feature_name, "psi_numeric", psi_value, len(ref), len(cur))


def psi_categorical(reference: pd.Series, current: pd.Series, feature_name: str) -> dict:
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        raise InsufficientDataError(feature_name, min(len(ref), len(cur)), 1)

    categories = list(ref.unique())
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()

    unseen_mass = cur_counts[~cur_counts.index.isin(categories)].sum()

    ref_pct = np.array([ref_counts.get(c, 0) for c in categories] + [0]) / len(ref)
    cur_pct = np.array(
        [cur_counts.get(c, 0) for c in categories] + [unseen_mass]
    ) / len(cur)

    psi_value = _psi_from_percents(ref_pct, cur_pct)
    result = _build_result(feature_name, "psi_categorical", psi_value, len(ref), len(cur))
    result["unseen_category_share"] = round(float(unseen_mass / len(cur)), 6)
    return result


def _build_result(feature_name: str, test: str, psi_value: float, n_ref: int, n_cur: int) -> dict:
    if psi_value < 0.10:
        severity = "none"
    elif psi_value < 0.25:
        severity = "moderate"
    else:
        severity = "major"
    return {
        "feature": feature_name,
        "test": test,
        "psi": round(psi_value, 6),
        "severity": severity,
        "drift_detected": psi_value >= 0.10,
        "n_reference": int(n_ref),
        "n_current": int(n_cur),
    }