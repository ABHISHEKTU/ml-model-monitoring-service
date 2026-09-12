"""API layer. Thin: validate input, call domain logic, map exceptions to HTTP."""

from __future__ import annotations
from app.drift.multivariate import multivariate_drift_test

import logging
from app.history_store import HistoryStore
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.drift.monitor import DriftMonitor
import numpy as np
from app.drift.ks_test import ks_drift_test
from app.drift.psi import psi_numeric
from app.exceptions import (
    InsufficientDataError,
    MonitoringError,
    ReferenceNotSetError,
    SchemaMismatchError,
)
from app.logging_config import setup_logging
from app.schemas import DriftCheckRequest, ReferenceUploadRequest, ReferenceUploadResponse
from app.storage import FileStore

setup_logging()
logger = logging.getLogger("model_monitor.api")

app = FastAPI(title="ML Model Monitoring Service", version="1.0.0")

store = FileStore()
history_store = HistoryStore()


@app.exception_handler(ReferenceNotSetError)
async def handle_no_reference(request: Request, exc: ReferenceNotSetError):
    return JSONResponse(status_code=404, content={"error": "reference_not_set", "detail": str(exc)})


@app.exception_handler(SchemaMismatchError)
async def handle_schema_mismatch(request: Request, exc: SchemaMismatchError):
    return JSONResponse(status_code=422, content={"error": "schema_mismatch", "detail": str(exc)})


@app.exception_handler(InsufficientDataError)
async def handle_insufficient_data(request: Request, exc: InsufficientDataError):
    return JSONResponse(status_code=422, content={"error": "insufficient_data", "detail": str(exc)})


@app.exception_handler(MonitoringError)
async def handle_generic_monitoring_error(request: Request, exc: MonitoringError):
    return JSONResponse(status_code=400, content={"error": "monitoring_error", "detail": str(exc)})


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Something went wrong."})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reference", response_model=ReferenceUploadResponse)
def upload_reference(payload: ReferenceUploadRequest):
    if not payload.numeric_features and not payload.categorical_features:
        raise HTTPException(status_code=422, detail="Specify at least one numeric or categorical feature.")

    df = pd.DataFrame.from_records(payload.records)
    missing = [f for f in payload.numeric_features + payload.categorical_features if f not in df.columns]
    if missing:
        raise SchemaMismatchError(missing=missing, extra=[])

    store.save(payload.model_id, df)
    store.save_config(payload.model_id, {
        "numeric_features": payload.numeric_features,
        "categorical_features": payload.categorical_features,
    })
    logger.info("Reference set for model_id=%s rows=%d", payload.model_id, len(df))

    if payload.reference_predictions is not None:
        store.save_predictions(payload.model_id, payload.reference_predictions)

    return ReferenceUploadResponse(
        model_id=payload.model_id, n_rows=len(df),
        numeric_features=payload.numeric_features,
        categorical_features=payload.categorical_features,
        message="Reference baseline stored.",
    )


@app.post("/monitor/drift")
def check_drift(payload: DriftCheckRequest):
    reference_df = store.load(payload.model_id)

    cfg = store.load_config(payload.model_id)  # raises ReferenceNotSetError if absent

    current_df = pd.DataFrame.from_records(payload.records)

    monitor = DriftMonitor(
        numeric_features=cfg["numeric_features"],
        categorical_features=cfg["categorical_features"],
        ks_alpha=payload.ks_alpha,
        psi_bins=payload.psi_bins,
    )
    report = monitor.run(reference_df, current_df)
    try:
        report["multivariate_drift"] = multivariate_drift_test(
            reference_df, current_df, cfg["numeric_features"], cfg["categorical_features"],
        )
    except InsufficientDataError as e:
        report["errors"].append({"feature": "multivariate", "reason": str(e)})
    logger.info("Drift check model_id=%s drifted=%s/%s", payload.model_id,
                report["n_features_drifted"], report["n_features_checked"])
    history_store.append(payload.model_id, report)
    if payload.current_predictions is not None and store.has_predictions(payload.model_id):
        try:
            ref_preds = store.load_predictions(payload.model_id)
            pred_ks = ks_drift_test(np.array(ref_preds), np.array(payload.current_predictions), "prediction")
            pred_psi = psi_numeric(np.array(ref_preds), np.array(payload.current_predictions), "prediction")
            report["prediction_drift"] = {"ks": pred_ks, "psi": pred_psi}
        except InsufficientDataError as e:
            report["errors"].append({"feature": "prediction", "reason": str(e)})
    return report


@app.get("/reference/{model_id}/exists")
def reference_exists(model_id: str):
    return {"model_id": model_id, "exists": store.exists(model_id)}


@app.get("/monitor/history/{model_id}")
def get_history(model_id: str, limit: int = 10):
    """Fetch the last `limit` drift reports for a model_id, most recent last."""
    return {"model_id": model_id, "history": history_store.load(model_id, limit=limit)}