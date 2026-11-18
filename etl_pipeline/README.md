# Credit Card Default Prediction (CCDP) ETL Pipeline

## Overview
An enterprise-grade **Extract, Transform, Load (ETL)** pipeline orchestrated using **Apache Airflow Directed Acyclic Graphs (DAGs)** (TaskFlow API) and a **Standalone Local Execution Runner** for the **Credit Card Default Prediction Dataset**.

- **Dataset Source**: [Kaggle Credit Card Defaulter Prediction (gauravtopre/credit-card-defaulter-prediction)](https://www.kaggle.com/datasets/gauravtopre/credit-card-defaulter-prediction)

The pipeline ingests raw client profile, repayment history, bill statement, and payment history data, applies systematic feature engineering (credit utilization, debt-to-bill ratios, repayment delay aggregates, outlier clipping), and loads the curated records into a PostgreSQL analytical warehouse table (`credit_card_default_data`) or local curated CSV storage.

---

## Architecture & Data Flow

```
+-------------------------------------------------------------+
| Kaggle API (gauravtopre/credit-card-defaulter-prediction)   |
| OR Local notebook/data/raw.csv                              |
| OR MySQL Database (credit table)                            |
+------------------------------+------------------------------+
                               |
                               v
       +--------------------------------------------------+
       |   Task 1: extract_credit_card_data               |
       |   - Ingest 30,000+ client default records        |
       +------------------------+-------------------------+
                                |
                                v
       +--------------------------------------------------+
       |   Task 2: transform_credit_card_data             |
       |   - Standardize column names & customer ID       |
       |   - Remap Education & Marriage categorical codes |
       |   - Compute Repayment Delay metrics (AVG/MAX)    |
       |   - Compute Bill & Payment Aggregates            |
       |   - Compute Credit Utilization & Bill Trend      |
       |   - Compute Pay-to-Bill Ratio & Age Groups       |
       |   - Apply IQR Outlier Capping & Imputation       |
       +------------------------+-------------------------+
                                |
                                v
       +--------------------------------------------------+
       |   Task 3: load_credit_card_data                  |
       |   - Connect to Postgres (PostgresHook)           |
       |   - Create table: credit_card_default_data       |
       |   - Idempotent upsert on conflict (customer_id)  |
       +--------------------------------------------------+
```

---

## Directory Structure

```
etl_pipeline/
├── dags/
│   ├── .airflowignore
│   └── etl_credit_card_default.py      # Main Airflow DAG using TaskFlow API
├── include/
│   ├── __init__.py
│   ├── logger.py                       # Timestamped logging to logs/ & console
│   ├── exception.py                    # CustomETLException with traceback
│   └── transformed_credit_card_default.csv # Local transformed dataset output
├── logs/                               # Auto-created execution logs
├── plugins/
│   └── __init__.py
├── tests/
│   └── dags/
│       └── test_dag_integrity.py       # DAG syntax and DagBag integrity tests
├── .dockerignore
├── .env.example                        # Template environment variables
├── airflow_settings.yaml               # Preconfigured connections and variables
├── docker-compose.yml                  # Postgres + Airflow Webserver + Scheduler
├── Dockerfile                          # Airflow custom container image
├── main.py                             # Standalone local ETL runner
├── packages.txt                        # OS-level packages (libpq-dev, etc.)
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation
```

---

## Feature Engineering Breakdown

The transformation layer calculates the following financial risk indicators:

| Feature Name | Description | Formula / Logic |
| :--- | :--- | :--- |
| `AVG_PAY_DELAY` | Mean repayment delay across 6 months | `mean(PAY_0, PAY_2, ..., PAY_6)` |
| `MAX_PAY_DELAY` | Worst delinquent delay status | `max(PAY_0, PAY_2, ..., PAY_6)` |
| `NUM_MONTHS_DELAYED` | Count of months with delayed payment | `sum(PAY_i > 0)` |
| `TOTAL_BILL_AMT` | Cumulative bill statement amount | `sum(BILL_AMT1 ... BILL_AMT6)` |
| `AVG_BILL_AMT` | Mean monthly statement amount | `mean(BILL_AMT1 ... BILL_AMT6)` |
| `TOTAL_PAY_AMT` | Cumulative payment amount | `sum(PAY_AMT1 ... PAY_AMT6)` |
| `AVG_PAY_AMT` | Mean monthly payment amount | `mean(PAY_AMT1 ... PAY_AMT6)` |
| `CREDIT_UTILIZATION` | Latest credit line usage ratio | `BILL_AMT1 / (LIMIT_BAL + 1e-5)` |
| `BILL_TREND` | Net debt trajectory (Recent vs Oldest) | `BILL_AMT1 - BILL_AMT6` |
| `PAY_TO_BILL_RATIO` | Cumulative payment to bill ratio | `TOTAL_PAY_AMT / (TOTAL_BILL_AMT + 1e-5)` |
| `AGE_GROUP` | Categorical demographic age bucket | Binned: `[20-30, 30-40, 40-50, 50-60, 60+]` |
| `EDUCATION` | Normalized education level | `[0, 5, 6] -> 4 (Others)` |
| `MARRIAGE` | Normalized marital status | `0 -> 3 (Others)` |

Continuous variables are bounded using interquartile range (IQR) clipping: $[Q_1 - 1.5 \times \text{IQR},\, Q_3 + 1.5 \times \text{IQR}]$.

---

## Quick Start Guide

### 1. Standalone Execution (No Docker Required)

You can run the ETL pipeline locally in any Python environment:

```bash
# Run standalone ETL pipeline
python etl_pipeline/main.py
```

This will:
- Extract records from `notebook/data/raw.csv` (or MySQL if configured)
- Run all data cleaning and feature engineering transformations
- Export `etl_pipeline/include/transformed_credit_card_default.csv`
- Log execution logs in `etl_pipeline/logs/`

### 2. Orchestrated Airflow Execution via Docker

To start the full Apache Airflow and PostgreSQL analytical stack:

```bash
# Navigate to etl_pipeline directory
cd etl_pipeline

# Copy sample environment variables
cp .env.example .env

# Launch Postgres + Airflow Webserver + Scheduler
docker-compose up -d
```

Access the Airflow UI at:
- **URL**: [http://localhost:8080](http://localhost:8080)
- **Username**: `airflow`
- **Password**: `airflow`
- **DAG ID**: `ccdp_credit_card_default_etl_pipeline`

---

## Testing & Validation

Run unit and DAG syntax tests using pytest:

```bash
pytest etl_pipeline/tests/dags/test_dag_integrity.py -v
```

---

## Downstream ML Pipeline Integration

The curated dataset generated by this ETL pipeline (`etl_pipeline/include/transformed_credit_card_default.csv`) is automatically consumed by the machine learning pipeline:

* **Consumer**: `src/CCDP/components/data_ingestion.py`
* **Triggering**: If the transformed CSV is not found, `data_ingestion.py` automatically invokes `run_standalone_etl()` from this package.
* **Model Training**: Runs 80/20 train-test splitting and passes the records to `src/CCDP/components/data_transformation.py` and `model_trainer.py` to train the production classification models.

