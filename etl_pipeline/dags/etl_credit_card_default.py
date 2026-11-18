"""
Apache Airflow Directed Acyclic Graph (DAG) for Credit Card Default Prediction (CCDP) ETL Pipeline
Orchestrated using Airflow TaskFlow API with PostgreSQL analytical storage
"""

from datetime import datetime, timedelta
import logging
import os
import sys
import numpy as np
import pandas as pd

# Try importing custom logger and exception handler from include package
try:
    from etl_pipeline.include.logger import logging as custom_logging, logger
    from etl_pipeline.include.exception import CustomETLException
except ImportError:
    try:
        from include.logger import logging as custom_logging, logger
        from include.exception import CustomETLException
    except ImportError:
        logger = logging.getLogger("ccdp_etl_dag")
        class CustomETLException(Exception):
            def __init__(self, error_message, error_details: sys):
                super().__init__(str(error_message))
                logger.error(f"ETL Error: {error_message}")

# Airflow imports with fallback compatibility
try:
    from airflow import DAG
    from airflow.decorators import task
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    from airflow.utils.timezone import datetime as tz_datetime
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    def task():
        def decorator(f):
            return f
        return decorator

POSTGRES_CONN_ID = os.getenv('POSTGRES_CONN_ID', 'postgres_default')
KAGGLE_DATASET = os.getenv('KAGGLE_DATASET', 'gauravtopre/credit-card-defaulter-prediction')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='ccdp_credit_card_default_etl_pipeline',
        default_args=default_args,
        schedule='@daily',
        start_date=tz_datetime(2026, 1, 1),
        catchup=False,
        tags=['ccdp', 'credit_card', 'default_prediction', 'etl', 'kaggle', 'finance']
    ) as dag:

        @task()
        def extract_credit_card_data():
            """
            Extract raw credit card client data from local storage, Kaggle API, or database.
            """
            logger.info(">>> [EXTRACT TASK START] Ingesting credit card client data...")
            try:
                candidate_paths = [
                    os.path.join(os.path.dirname(__file__), '..', '..', 'notebook', 'data', 'raw.csv'),
                    os.path.join(os.path.dirname(__file__), '..', 'include', 'raw.csv'),
                    os.path.join('notebook', 'data', 'raw.csv'),
                    os.path.join('artifacts', 'raw.csv'),
                    '/opt/airflow/notebook/data/raw.csv',
                    '/opt/airflow/include/raw.csv'
                ]

                df = None
                for path in candidate_paths:
                    if os.path.exists(path):
                        logger.info(f"Extracting raw client dataset from local path: {path}")
                        df = pd.read_csv(path)
                        break

                # Automated Kaggle Dataset Download Fallback
                if df is None:
                    try:
                        logger.info(f"Local file not found. Attempting extraction from Kaggle: [{KAGGLE_DATASET}]...")
                        import kaggle
                        download_dir = os.path.join(os.path.dirname(__file__), '..', 'include')
                        os.makedirs(download_dir, exist_ok=True)
                        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=download_dir, unzip=True)
                        
                        # Find downloaded CSV in include directory
                        for file in os.listdir(download_dir):
                            if file.endswith('.csv') and 'transformed' not in file:
                                kaggle_csv = os.path.join(download_dir, file)
                                logger.info(f"Loaded Kaggle dataset from: {kaggle_csv}")
                                df = pd.read_csv(kaggle_csv)
                                break
                    except Exception as kaggle_err:
                        logger.warning(f"Kaggle API extraction fallback unavailable or failed: {kaggle_err}")

                if df is None or df.empty:
                    # Optional database fallback
                    try:
                        import pymysql
                        host = os.getenv("host")
                        user = os.getenv("user")
                        password = os.getenv("password")
                        db = os.getenv('db')
                        if host and user:
                            logger.info("Attempting extraction from MySQL database...")
                            mydb = pymysql.connect(host=host, user=user, password=password, db=db)
                            df = pd.read_sql_query('SELECT * FROM credit', mydb)
                    except Exception as db_err:
                        logger.warning(f"Database extraction fallback failed: {db_err}")

                if df is None or df.empty:
                    raise CustomETLException("No source credit card dataset could be found or extracted.", sys)

                logger.info(f">>> [EXTRACT TASK COMPLETE] Successfully extracted {len(df)} records with columns: {list(df.columns)}")
                return df.to_dict(orient='records')

            except Exception as e:
                logger.error(f"Critical error in extract_credit_card_data: {str(e)}")
                raise CustomETLException(e, sys)

        @task()
        def transform_credit_card_data(raw_records):
            """
            Perform data cleaning, categorical remapping, debt & repayment ratio engineering,
            and outlier capping.
            """
            logger.info(">>> [TRANSFORM TASK START] Performing feature engineering & cleaning...")
            try:
                if not raw_records:
                    raise CustomETLException("Received empty records in transform task.", sys)

                df = pd.DataFrame(raw_records)
                logger.info(f"Initial raw dataset shape: {df.shape}")

                # Standardize column headers
                df.columns = [c.strip().upper().replace('.', '_') for c in df.columns]

                # Identify customer identifier
                if 'ID' in df.columns:
                    df.rename(columns={'ID': 'CUSTOMER_ID'}, inplace=True)
                elif 'CUSTOMER_ID' not in df.columns:
                    df['CUSTOMER_ID'] = range(1, len(df) + 1)

                # Standardize Target Column Name
                target_col = 'DEFAULT_PAYMENT_NEXT_MONTH'
                if target_col not in df.columns:
                    possible_targets = [c for c in df.columns if 'DEFAULT' in c]
                    if possible_targets:
                        df.rename(columns={possible_targets[0]: target_col}, inplace=True)

                # Categorical Remapping
                if 'EDUCATION' in df.columns:
                    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
                if 'MARRIAGE' in df.columns:
                    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)

                # Feature Engineering: Repayment Delays
                pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6] if f'PAY_{i}' in df.columns]
                if pay_cols:
                    logger.info("Engineering repayment delay features...")
                    df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
                    df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
                    df['NUM_MONTHS_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)

                # Feature Engineering: Bill Statements & Payments
                bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
                pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]

                if bill_cols:
                    logger.info("Engineering bill amount aggregations...")
                    df['TOTAL_BILL_AMT'] = df[bill_cols].sum(axis=1)
                    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
                    if 'LIMIT_BAL' in df.columns and 'BILL_AMT1' in df.columns:
                        df['CREDIT_UTILIZATION'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1e-5)
                    if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
                        df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']

                if pay_amt_cols:
                    logger.info("Engineering payment amount aggregations...")
                    df['TOTAL_PAY_AMT'] = df[pay_amt_cols].sum(axis=1)
                    df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)

                if 'TOTAL_PAY_AMT' in df.columns and 'TOTAL_BILL_AMT' in df.columns:
                    df['PAY_TO_BILL_RATIO'] = df['TOTAL_PAY_AMT'] / (df['TOTAL_BILL_AMT'] + 1e-5)

                if 'AGE' in df.columns:
                    bins = [20, 30, 40, 50, 60, 120]
                    labels = [0, 1, 2, 3, 4]
                    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, include_lowest=True).astype(int)

                # Outlier clipping via IQR bounds
                columns_to_cap = ['LIMIT_BAL', 'TOTAL_BILL_AMT', 'TOTAL_PAY_AMT',
                                  'CREDIT_UTILIZATION', 'BILL_TREND', 'AVG_PAY_DELAY', 'MAX_PAY_DELAY']
                for col in columns_to_cap:
                    if col in df.columns:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

                # Missing Value Imputation
                nan_count = df.isnull().sum().sum()
                if nan_count > 0:
                    logger.info(f"Imputing {nan_count} missing values...")
                    df.bfill(inplace=True)
                    df.ffill(inplace=True)

                transformed_records = df.to_dict(orient='records')
                logger.info(f">>> [TRANSFORM TASK COMPLETE] Transformed {len(transformed_records)} records with {len(df.columns)} features.")
                return transformed_records

            except Exception as e:
                logger.error(f"Critical error in transform_credit_card_data: {str(e)}")
                raise CustomETLException(e, sys)

        @task()
        def load_credit_card_data(transformed_records):
            """
            Load transformed records into PostgreSQL analytical database table.
            """
            logger.info(">>> [LOAD TASK START] Initializing PostgreSQL loading...")
            try:
                if not transformed_records:
                    logger.warning("No transformed records to load into database.")
                    return

                pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
                logger.info(f"Connected to PostgreSQL via [{POSTGRES_CONN_ID}].")

                create_table_sql = """
                CREATE TABLE IF NOT EXISTS credit_card_default_data (
                    customer_id BIGINT PRIMARY KEY,
                    limit_bal FLOAT,
                    sex INT,
                    education INT,
                    marriage INT,
                    age INT,
                    age_group INT,
                    pay_0 INT,
                    pay_2 INT,
                    pay_3 INT,
                    pay_4 INT,
                    pay_5 INT,
                    pay_6 INT,
                    bill_amt1 FLOAT,
                    bill_amt2 FLOAT,
                    bill_amt3 FLOAT,
                    bill_amt4 FLOAT,
                    bill_amt5 FLOAT,
                    bill_amt6 FLOAT,
                    pay_amt1 FLOAT,
                    pay_amt2 FLOAT,
                    pay_amt3 FLOAT,
                    pay_amt4 FLOAT,
                    pay_amt5 FLOAT,
                    pay_amt6 FLOAT,
                    avg_pay_delay FLOAT,
                    max_pay_delay FLOAT,
                    num_months_delayed INT,
                    total_bill_amt FLOAT,
                    avg_bill_amt FLOAT,
                    total_pay_amt FLOAT,
                    avg_pay_amt FLOAT,
                    credit_utilization FLOAT,
                    bill_trend FLOAT,
                    pay_to_bill_ratio FLOAT,
                    default_payment_next_month INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                pg_hook.run(create_table_sql)
                logger.info("PostgreSQL table credit_card_default_data verified/created.")

                insert_sql = """
                INSERT INTO credit_card_default_data (
                    customer_id, limit_bal, sex, education, marriage, age, age_group,
                    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                    bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
                    pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6,
                    avg_pay_delay, max_pay_delay, num_months_delayed,
                    total_bill_amt, avg_bill_amt, total_pay_amt, avg_pay_amt,
                    credit_utilization, bill_trend, pay_to_bill_ratio, default_payment_next_month
                ) VALUES (
                    %(customer_id)s, %(limit_bal)s, %(sex)s, %(education)s, %(marriage)s, %(age)s, %(age_group)s,
                    %(pay_0)s, %(pay_2)s, %(pay_3)s, %(pay_4)s, %(pay_5)s, %(pay_6)s,
                    %(bill_amt1)s, %(bill_amt2)s, %(bill_amt3)s, %(bill_amt4)s, %(bill_amt5)s, %(bill_amt6)s,
                    %(pay_amt1)s, %(pay_amt2)s, %(pay_amt3)s, %(pay_amt4)s, %(pay_amt5)s, %(pay_amt6)s,
                    %(avg_pay_delay)s, %(max_pay_delay)s, %(num_months_delayed)s,
                    %(total_bill_amt)s, %(avg_bill_amt)s, %(total_pay_amt)s, %(avg_pay_amt)s,
                    %(credit_utilization)s, %(bill_trend)s, %(pay_to_bill_ratio)s, %(default_payment_next_month)s
                )
                ON CONFLICT (customer_id) DO UPDATE SET
                    limit_bal = EXCLUDED.limit_bal,
                    sex = EXCLUDED.sex,
                    education = EXCLUDED.education,
                    marriage = EXCLUDED.marriage,
                    age = EXCLUDED.age,
                    age_group = EXCLUDED.age_group,
                    pay_0 = EXCLUDED.pay_0,
                    pay_2 = EXCLUDED.pay_2,
                    pay_3 = EXCLUDED.pay_3,
                    pay_4 = EXCLUDED.pay_4,
                    pay_5 = EXCLUDED.pay_5,
                    pay_6 = EXCLUDED.pay_6,
                    bill_amt1 = EXCLUDED.bill_amt1,
                    bill_amt2 = EXCLUDED.bill_amt2,
                    bill_amt3 = EXCLUDED.bill_amt3,
                    bill_amt4 = EXCLUDED.bill_amt4,
                    bill_amt5 = EXCLUDED.bill_amt5,
                    bill_amt6 = EXCLUDED.bill_amt6,
                    pay_amt1 = EXCLUDED.pay_amt1,
                    pay_amt2 = EXCLUDED.pay_amt2,
                    pay_amt3 = EXCLUDED.pay_amt3,
                    pay_amt4 = EXCLUDED.pay_amt4,
                    pay_amt5 = EXCLUDED.pay_amt5,
                    pay_amt6 = EXCLUDED.pay_amt6,
                    avg_pay_delay = EXCLUDED.avg_pay_delay,
                    max_pay_delay = EXCLUDED.max_pay_delay,
                    num_months_delayed = EXCLUDED.num_months_delayed,
                    total_bill_amt = EXCLUDED.total_bill_amt,
                    avg_bill_amt = EXCLUDED.avg_bill_amt,
                    total_pay_amt = EXCLUDED.total_pay_amt,
                    avg_pay_amt = EXCLUDED.avg_pay_amt,
                    credit_utilization = EXCLUDED.credit_utilization,
                    bill_trend = EXCLUDED.bill_trend,
                    pay_to_bill_ratio = EXCLUDED.pay_to_bill_ratio,
                    default_payment_next_month = EXCLUDED.default_payment_next_month;
                """

                count = 0
                for r in transformed_records:
                    row_params = {
                        'customer_id': int(r.get('CUSTOMER_ID', 0)),
                        'limit_bal': float(r.get('LIMIT_BAL', 0.0)),
                        'sex': int(r.get('SEX', 0)),
                        'education': int(r.get('EDUCATION', 0)),
                        'marriage': int(r.get('MARRIAGE', 0)),
                        'age': int(r.get('AGE', 0)),
                        'age_group': int(r.get('AGE_GROUP', 0)),
                        'pay_0': int(r.get('PAY_0', 0)),
                        'pay_2': int(r.get('PAY_2', 0)),
                        'pay_3': int(r.get('PAY_3', 0)),
                        'pay_4': int(r.get('PAY_4', 0)),
                        'pay_5': int(r.get('PAY_5', 0)),
                        'pay_6': int(r.get('PAY_6', 0)),
                        'bill_amt1': float(r.get('BILL_AMT1', 0.0)),
                        'bill_amt2': float(r.get('BILL_AMT2', 0.0)),
                        'bill_amt3': float(r.get('BILL_AMT3', 0.0)),
                        'bill_amt4': float(r.get('BILL_AMT4', 0.0)),
                        'bill_amt5': float(r.get('BILL_AMT5', 0.0)),
                        'bill_amt6': float(r.get('BILL_AMT6', 0.0)),
                        'pay_amt1': float(r.get('PAY_AMT1', 0.0)),
                        'pay_amt2': float(r.get('PAY_AMT2', 0.0)),
                        'pay_amt3': float(r.get('PAY_AMT3', 0.0)),
                        'pay_amt4': float(r.get('PAY_AMT4', 0.0)),
                        'pay_amt5': float(r.get('PAY_AMT5', 0.0)),
                        'pay_amt6': float(r.get('PAY_AMT6', 0.0)),
                        'avg_pay_delay': float(r.get('AVG_PAY_DELAY', 0.0)),
                        'max_pay_delay': float(r.get('MAX_PAY_DELAY', 0.0)),
                        'num_months_delayed': int(r.get('NUM_MONTHS_DELAYED', 0)),
                        'total_bill_amt': float(r.get('TOTAL_BILL_AMT', 0.0)),
                        'avg_bill_amt': float(r.get('AVG_BILL_AMT', 0.0)),
                        'total_pay_amt': float(r.get('TOTAL_PAY_AMT', 0.0)),
                        'avg_pay_amt': float(r.get('AVG_PAY_AMT', 0.0)),
                        'credit_utilization': float(r.get('CREDIT_UTILIZATION', 0.0)),
                        'bill_trend': float(r.get('BILL_TREND', 0.0)),
                        'pay_to_bill_ratio': float(r.get('PAY_TO_BILL_RATIO', 0.0)),
                        'default_payment_next_month': int(r.get('DEFAULT_PAYMENT_NEXT_MONTH', 0))
                    }
                    pg_hook.run(insert_sql, parameters=row_params)
                    count += 1

                logger.info(f">>> [LOAD TASK COMPLETE] Successfully loaded/upserted {count} records into PostgreSQL.")

            except Exception as e:
                logger.error(f"Critical error in load_credit_card_data: {str(e)}")
                raise CustomETLException(e, sys)

        # Pipeline DAG workflow definition
        raw_data = extract_credit_card_data()
        transformed_data = transform_credit_card_data(raw_data)
        load_credit_card_data(transformed_data)
