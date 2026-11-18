"""
DAG integrity and validation tests for Credit Card Default Prediction (CCDP) ETL Pipeline
"""

import os
import logging
from contextlib import contextmanager
import pytest

try:
    from airflow.models import DagBag
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


@contextmanager
def suppress_logging(namespace):
    logger = logging.getLogger(namespace)
    old_value = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = old_value


def test_dag_file_syntax():
    """Verify that the DAG python file compiles without syntax errors."""
    dag_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dags', 'etl_credit_card_default.py')
    with open(dag_path, 'r', encoding='utf-8') as f:
        source = f.read()
    compiled = compile(source, dag_path, 'exec')
    assert compiled is not None


@pytest.mark.skipif(not AIRFLOW_AVAILABLE, reason="Airflow package not installed in current environment")
def test_airflow_dag_bag():
    """Ensure no import errors exist when Airflow loads the DAG."""
    dags_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dags')
    with suppress_logging("airflow"):
        dag_bag = DagBag(dag_folder=dags_dir, include_examples=False)

    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"
    assert 'ccdp_credit_card_default_etl_pipeline' in dag_bag.dags
