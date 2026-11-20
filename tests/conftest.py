"""
conftest.py — Pytest configuration for the CCDP test suite.

Ensures:
  - Project root is on sys.path so `src.CCDP` and `application` are importable.
  - MLflow noise is suppressed in CI.
  - artifacts/model.pkl must exist before tests run (trained by `python main.py`).
"""
import os
import sys
import pytest

# ── Make project root importable ──────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Suppress MLflow agent-hint stderr noise ────────────────────────────────────
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "")


# ── Guard: fail fast if model artifacts are missing ───────────────────────────
def pytest_configure(config):
    model_path = os.path.join(ROOT, "artifacts", "model.pkl")
    preprocessor_path = os.path.join(ROOT, "artifacts", "preprocessor.pkl")
    missing = [p for p in (model_path, preprocessor_path) if not os.path.exists(p)]
    if missing:
        pytest.exit(
            f"\n[conftest] Missing required artifacts:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nRun `python main.py` first to train the model and generate artifacts.\n",
            returncode=1,
        )
