from __future__ import annotations

from pydantic import BaseModel, Field


class ReferenceUploadRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str = Field(..., min_length=1)
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    records: list[dict] = Field(..., min_length=1)
    reference_predictions: list[float] | None = Field(default=None, description="Model's predictions on the reference/training batch")


class ReferenceUploadResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str
    n_rows: int
    numeric_features: list[str]
    categorical_features: list[str]
    message: str


class DriftCheckRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: str = Field(..., min_length=1)
    records: list[dict] = Field(..., min_length=1)
    ks_alpha: float = Field(default=0.05, gt=0, lt=1)
    psi_bins: int = Field(default=10, ge=2, le=50)
    current_predictions: list[float] | None = Field(default=None, description="Model's predictions on the current/live batch")