"""
Quick script to test multivariate drift detection.
Scenario: reference has age & income POSITIVELY correlated (older -> more income).
Current has the SAME marginal ranges for age and income individually (so per-feature
KS/PSI should stay clean) but the correlation is FLIPPED (older -> less income).
This is the exact case marginal tests can't catch but a domain classifier can.
"""
import numpy as np
import requests

np.random.seed(1)
BASE_URL = "http://127.0.0.1:8000"

ref_age = np.random.uniform(20, 60, 200)
ref_income = 20000 + ref_age * 800 + np.random.normal(0, 3000, 200)

ref_records = [{"age": float(a), "income": float(i)} for a, i in zip(ref_age, ref_income)]

resp = requests.post(f"{BASE_URL}/reference", json={
    "model_id": "corr_test",
    "numeric_features": ["age", "income"],
    "categorical_features": [],
    "records": ref_records,
})
print("Reference upload:", resp.status_code, resp.json())

cur_age = np.random.uniform(20, 60, 200)
cur_income = 20000 + (80 - cur_age) * 800 + np.random.normal(0, 3000, 200)

cur_records = [{"age": float(a), "income": float(i)} for a, i in zip(cur_age, cur_income)]

resp = requests.post(f"{BASE_URL}/monitor/drift", json={
    "model_id": "corr_test",
    "records": cur_records,
})
result = resp.json()
print("\n--- Per-feature results (expect: clean / no drift) ---")
for r in result["results"]:
    print(r["feature"], "->", r.get("ks", {}).get("drift_detected"), r.get("psi", {}).get("drift_detected"))

print("\n--- Multivariate result (expect: drift_detected True) ---")
print(result.get("multivariate_drift"))