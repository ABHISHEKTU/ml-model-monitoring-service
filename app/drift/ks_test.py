"""
Kolmogorov-Smirnov two-sample test.

CONCEPT:
- ECDF(x) = fraction of points <= x, for a sorted sample. A staircase from 0 to 1.
- KS statistic D = max vertical gap between ECDF_reference(x) and ECDF_current(x)
  over all x. Distribution-free — no normality assumption, catches ANY difference
  in shape/location/scale, not just a mean shift.
- p-value = probability of seeing a gap this large if both samples truly came from
  the SAME distribution (null hypothesis H0). p < alpha (usually 0.05) => reject H0
  => distributions differ significantly => DRIFT.
- Limitation: sensitive to sample size — with huge batches, even tiny/noise-level
  differences become "significant". That's why we pair this with PSI later, which
  measures magnitude instead of pure significance.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from app.exceptions import InsufficientDataError

MIN_SAMPLES = 20  # below this, KS p-values are unreliable/noisy


def ks_drift_test(reference: np.ndarray, current: np.ndarray, feature_name: str,
                   alpha: float = 0.05) -> dict:
    ref = np.asarray(reference, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = np.asarray(current, dtype=float)
    cur = cur[~np.isnan(cur)]

    if len(ref) < MIN_SAMPLES:
        raise InsufficientDataError(feature_name, len(ref), MIN_SAMPLES)
    if len(cur) < MIN_SAMPLES:
        raise InsufficientDataError(feature_name, len(cur), MIN_SAMPLES)

    statistic, p_value = stats.ks_2samp(ref, cur)

    return {
        "feature": feature_name,
        "test": "ks_2samp",
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "drift_detected": bool(p_value < alpha),
        "n_reference": int(len(ref)),
        "n_current": int(len(cur)),
    }