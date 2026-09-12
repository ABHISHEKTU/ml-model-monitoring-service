"""
DriftMonitor: orchestrates per-feature drift checks across a whole dataframe.
Stateless per call — reference + current passed in each time, storage is a
separate concern. Fail-soft: one bad feature never crashes the whole report.
"""

from __future__ import annotations

import logging

import pandas as pd

from app.drift.ks_test import ks_drift_test
from app.drift.psi import psi_categorical, psi_numeric
from app.exceptions import InsufficientDataError, SchemaMismatchError

logger = logging.getLogger("model_monitor.drift")


class DriftMonitor:
    def __init__(self, numeric_features: list[str], categorical_features: list[str],
                 ks_alpha: float = 0.05, psi_bins: int = 10):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.ks_alpha = ks_alpha
        self.psi_bins = psi_bins

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required = set(self.numeric_features) | set(self.categorical_features)
        present = set(df.columns)
        missing = sorted(required - present)
        if missing:
            raise SchemaMismatchError(missing=missing, extra=[])

    def run(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        self._validate_schema(reference_df)
        self._validate_schema(current_df)

        results = []
        errors = []

        for feature in self.numeric_features:
            try:
                ks_res = ks_drift_test(
                    reference_df[feature].values, current_df[feature].values,
                    feature, alpha=self.ks_alpha,
                )
                psi_res = psi_numeric(
                    reference_df[feature].values, current_df[feature].values,
                    feature, bins=self.psi_bins,
                )
                results.append({"feature": feature, "type": "numeric",
                                 "ks": ks_res, "psi": psi_res})
            except InsufficientDataError as e:
                logger.warning("Skipping feature '%s': %s", feature, e)
                errors.append({"feature": feature, "reason": str(e)})
            except Exception as e:  # noqa: BLE001 - deliberate: never crash whole report
                logger.exception("Unexpected error on feature '%s'", feature)
                errors.append({"feature": feature, "reason": f"unexpected: {e}"})

        for feature in self.categorical_features:
            try:
                psi_res = psi_categorical(
                    reference_df[feature], current_df[feature], feature,
                )
                results.append({"feature": feature, "type": "categorical", "psi": psi_res})
            except InsufficientDataError as e:
                logger.warning("Skipping feature '%s': %s", feature, e)
                errors.append({"feature": feature, "reason": str(e)})
            except Exception as e:  # noqa: BLE001
                logger.exception("Unexpected error on feature '%s'", feature)
                errors.append({"feature": feature, "reason": f"unexpected: {e}"})

        n_drifted = sum(
            1 for r in results
            if r.get("psi", {}).get("drift_detected") or r.get("ks", {}).get("drift_detected")
        )

        return {
            "n_features_checked": len(results),
            "n_features_drifted": n_drifted,
            "overall_drift_detected": n_drifted > 0,
            "results": results,
            "errors": errors,
        }