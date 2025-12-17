
"""
End-to-end tests for the Loan Default Risk FastAPI service.

This test suite:
- Ensures the package is importable (adds project root to sys.path).
- Creates app/__init__.py if missing (so 'app' is a proper package).
- Exercises /health, /predict, /metrics, /monitor/drift, /monitor/model, /feedback.
- Generates a baseline_stats.json when absent and re-tests drift.

IMPORTANT:
- Run from the project root (the folder that contains 'app' and 'artifacts'):
  C:\Users\MEridu\OneDrive - Plan International\Desktop\Codiing UCU\env\Assignment_data sciencelife\dataset

Command:
  python -m pytest -q

If you prefer plain 'pytest -q', make sure your current directory is the project root.
"""

import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
import pytest

# ========= 1) Ensure the project root is on sys.path =========
PROJECT_ROOT = r"C:\Users\MEridu\OneDrive - Plan International\Desktop\Codiing UCU\env\Assignment_data sciencelife\dataset"
if not os.path.isdir(PROJECT_ROOT):
    # Fallback to relative path if the hard-coded path changes
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========= 2) Ensure 'app' is a proper package =========
APP_DIR = os.path.join(PROJECT_ROOT, "app")
INIT_FILE = os.path.join(APP_DIR, "__init__.py")
if not os.path.exists(INIT_FILE):
    # Create an empty __init__.py so Python treats 'app' as a package
    os.makedirs(APP_DIR, exist_ok=True)
    with open(INIT_FILE, "w", encoding="utf-8") as f:
        f.write("# Package marker\n")

# ========= 3) Now we can import the app and helpers =========
from fastapi.testclient import TestClient
from app.main import app
from app.settings import (
    FEATURE_ORDER,
    BASELINE_STATS_PATH,
    ARTIFACTS_DIR,
)
from app.model_loader import get_model, to_dataframe
from app.monitor import (
    compute_baseline_stats,
    save_baseline,
    load_baseline,
)

client = TestClient(app)


# ========= Utilities: sample payloads matching your FEATURE_ORDER =========
def _random_record() -> dict:
    """
    Build one synthetic borrower record compatible with FEATURE_ORDER and FastAPI schema.
    NOTE: The API schema expects aliases for age features (age_band_18_30, age_band_31_59),
    which map onto columns with hyphens internally.
    """
    # Ordinal bands as midpoints for simplicity
    income = random.choice([0, 1, 2, 3])             # <=300k, 300k-800k, 800k-1.5m, 1.5m-3m (example ordinals)
    loan_amt = random.choice([0, 1, 2, 3, 4])        # <=500k, 500k-1m, 1m-2m, 2m-3m, >3m
    empl_len = random.choice([0, 1, 2, 3, 4])        # <=1y, 1-3y, 3-5y, 5-10y, >10y
    dti = random.choice([0, 1, 2, 3, 4, 5])          # <=0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, >1.0

    # One-hot flags (simple consistent assignment)
    gender_f = float(random.choice([0, 1]))
    gender_m = float(1 - gender_f)
    age_31_59 = float(random.choice([0, 1]))
    age_18_30 = float(1 - age_31_59)
    own = float(random.choice([0, 1]))
    rent = float(random.choice([0, 1 - int(own)]))
    family = float(1 - int(own) - int(rent))
    # Keep them non-negative and sum <= 1
    if family < 0:
        family = 0.0

    return {
        "credit_score": float(random.randint(350, 820)),  # plausible range
        "previous_defaults": float(random.choice([0, 1, 2])),
        "loan_term_months": float(random.choice([6, 9, 12, 18, 24])),
        "income_band_ord": float(income),
        "loan_amount_band_ord": float(loan_amt),
        "employment_length_band_ord": float(empl_len),
        "dti_band_ord": float(dti),
        "gender_Female": gender_f,
        "gender_Male": gender_m,
        "age_band_18_30": age_18_30,     # <-- alias key used in JSON
        "age_band_31_59": age_31_59,     # <-- alias key used in JSON
        "home_ownership_Family": family,
        "home_ownership_Own": own,
        "home_ownership_Rent": rent,
    }


def _batch(n=5) -> list[dict]:
    return [_random_record() for _ in range(n)]


# ========= 4) Fixtures =========
@pytest.fixture(scope="session")
def model_exists():
    """Skip predict tests if model artifact is missing."""
    from app.settings import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found: {MODEL_PATH}. Train and copy best_hgb.pkl into artifacts/ first.")
    return True


@pytest.fixture(scope="session")
def ensure_baseline():
    """
    Ensure baseline_stats.json exists; if not, generate a baseline from a small synthetic batch.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    baseline = load_baseline()
    if baseline is None:
        df_ref = to_dataframe(_batch(200))  # 200 synthetic records
        stats = compute_baseline_stats(df_ref.astype("float32"))
        save_baseline(stats)
    return True


# ========= 5) Tests =========
def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "OK"


def test_metrics_endpoint_ok():
    r = client.get("/metrics")
    # Prometheus default content type
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")


def test_predict_single_ok(model_exists):
    payload = {"records": [_random_record()]}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "probabilities" in data and "predictions" in data and "threshold" in data
    assert len(data["probabilities"]) == 1
    assert len(data["predictions"]) == 1
    assert 0.0 <= data["probabilities"][0] <= 1.0
    assert data["predictions"][0] in [0, 1]


def test_predict_batch_ok(model_exists):
    payload = {"records": _batch(10)}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["probabilities"]) == 10
    assert len(data["predictions"]) == 10


def test_monitor_drift_create_then_report(ensure_baseline):
    # With a baseline present, drift report should return feature-wise PSI/KS
    payload = _batch(25)
    r = client.post("/monitor/drift", json={"records": payload})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert "features" in rep
    # Ensure all monitored features are listed
    assert set(rep["features"].keys()) == set(FEATURE_ORDER)


def test_model_monitor_insufficient_labels_initial():
    r = client.get("/monitor/model")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ["INSUFFICIENT_LABELS", "OK", "WARN", "ALERT"]


def test_feedback_and_model_monitor_progress(model_exists):
    """
    Push a few feedback points (simulated) and ensure the endpoint remains healthy.
    Note: Without many labels, status may still be 'INSUFFICIENT_LABELS'—this is fine.
    """
    # Call predict to populate recent scores
    payload = {"records": _batch(20)}
    r_pred = client.post("/predict", json=payload)
    assert r_pred.status_code == 200
    probs = r_pred.json()["probabilities"]

    # Submit feedback: pair some probs with synthetic labels
    for p in probs[:10]:
        label = random.choice([0, 1])
        r_fb = client.post(f"/feedback?prob={p}&label={label}")
        assert r_fb.status_code == 200

    r_mon = client.get("/monitor/model")
    assert r_mon.status_code == 200
    data = r_mon.json()
    assert data["status"] in ["INSUFFICIENT_LABELS", "OK", "WARN", "ALERT"]
    # If    # If enough labels accumulated, we expect rolling_pr_auc key
    if data["status"] != "INSUFFICIENT_LABELS":
