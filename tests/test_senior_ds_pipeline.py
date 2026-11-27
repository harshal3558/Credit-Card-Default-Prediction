import json
import os
import numpy as np
import pandas as pd

from application import app
from src.CCDP.components.data_transformation import (
    CorrelationSelector,
    FeatureEngineering,
    MutualInfoSelector,
)
from src.CCDP.components.model_monitering import ModelMonitor


def test_feature_name_propagation():
    """Verify FeatureEngineering -> CorrelationSelector -> MutualInfoSelector preserve exact names."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "LIMIT_BAL": rng.integers(10000, 200000, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(22, 60, n),
        "PAY_0": rng.integers(-1, 4, n),
        "PAY_2": rng.integers(-1, 4, n),
        "PAY_3": rng.integers(-1, 4, n),
        "PAY_4": rng.integers(-1, 4, n),
        "PAY_5": rng.integers(-1, 4, n),
        "PAY_6": rng.integers(-1, 4, n),
        "BILL_AMT1": rng.integers(0, 50000, n),
        "BILL_AMT2": rng.integers(0, 50000, n),
        "BILL_AMT3": rng.integers(0, 50000, n),
        "BILL_AMT4": rng.integers(0, 50000, n),
        "BILL_AMT5": rng.integers(0, 50000, n),
        "BILL_AMT6": rng.integers(0, 50000, n),
        "PAY_AMT1": rng.integers(0, 10000, n),
        "PAY_AMT2": rng.integers(0, 10000, n),
        "PAY_AMT3": rng.integers(0, 10000, n),
        "PAY_AMT4": rng.integers(0, 10000, n),
        "PAY_AMT5": rng.integers(0, 10000, n),
        "PAY_AMT6": rng.integers(0, 10000, n),
    })
    y = rng.integers(0, 2, n)

    fe = FeatureEngineering()
    X_fe = fe.fit_transform(df)
    fe_names = fe.get_feature_names_out()
    assert len(fe_names) == X_fe.shape[1]

    cs = CorrelationSelector(threshold=0.90)
    X_cs = cs.fit_transform(X_fe)
    cs_names = cs.get_feature_names_out(fe_names)
    assert len(cs_names) == X_cs.shape[1]

    mi = MutualInfoSelector(k=15, random_state=42)
    X_mi = mi.fit_transform(X_cs, y)
    mi_names = mi.get_feature_names_out(cs_names)
    assert len(mi_names) == X_mi.shape[1] == 15


def test_model_monitor_psi_and_ks():
    """Verify ModelMonitor reports OK on matching distributions and CRITICAL on shifted data."""
    rng = np.random.default_rng(7)
    X_ref = rng.normal(0.0, 1.0, size=(500, 4))
    X_same = rng.normal(0.0, 1.0, size=(500, 4))
    X_shifted = rng.normal(3.0, 1.0, size=(500, 4))

    monitor = ModelMonitor()
    ok_report = monitor.evaluate_drift(X_ref, X_same, feature_names=["f1", "f2", "f3", "f4"], save_report=False)
    assert ok_report["overall_status"] == "OK"

    crit_report = monitor.evaluate_drift(X_ref, X_shifted, feature_names=["f1", "f2", "f3", "f4"], save_report=False)
    assert crit_report["overall_status"] == "CRITICAL"
    assert len(crit_report["critical_features"]) == 4


def test_health_and_monitoring_endpoints():
    """Verify /health and /monitoring endpoints return valid Senior DS governance & drift payloads."""
    client = app.test_client()

    health_res = client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.get_json()
    assert health_data["status"] == "healthy"

    mon_res = client.get("/monitoring")
    assert mon_res.status_code == 200
    mon_data = mon_res.get_json()
    assert "governance_report" in mon_data
    assert "drift_report" in mon_data
    assert "business_cost_optimization" in mon_data["governance_report"]
    assert "fairness_audit" in mon_data["governance_report"]
