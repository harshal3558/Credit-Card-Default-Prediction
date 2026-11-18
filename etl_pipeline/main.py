"""
Standalone runner for Credit Card Default Prediction (CCDP) ETL Pipeline with custom logging & exception handling
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure etl_pipeline can be imported from root or etl_pipeline directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from etl_pipeline.include.logger import logger
    from etl_pipeline.include.exception import CustomETLException
except ImportError:
    from include.logger import logger
    from include.exception import CustomETLException


def run_standalone_etl():
    logger.info("======================================================================")
    logger.info(">>> Starting Credit Card Default Prediction ETL Pipeline (Standalone Mode)")
    logger.info("======================================================================")

    try:
        # 1. EXTRACT
        logger.info("Step 1: Extracting raw Credit Card Client Data...")
        raw_candidates = [
            os.path.join(os.path.dirname(__file__), '..', 'notebook', 'data', 'raw.csv'),
            os.path.join('notebook', 'data', 'raw.csv'),
            os.path.join('artifacts', 'raw.csv'),
            os.path.join(os.path.dirname(__file__), 'include', 'raw.csv')
        ]

        df = None
        for path in raw_candidates:
            if os.path.exists(path):
                logger.info(f"Loading raw credit card data from: {path}")
                df = pd.read_csv(path)
                break

        # Fallback to automated Kaggle dataset download
        kaggle_dataset = os.getenv('KAGGLE_DATASET', 'gauravtopre/credit-card-defaulter-prediction')
        if df is None or df.empty:
            try:
                logger.info(f"Local file not found. Attempting extraction from Kaggle dataset: [{kaggle_dataset}]...")
                import kaggle
                download_dir = os.path.join(os.path.dirname(__file__), 'include')
                os.makedirs(download_dir, exist_ok=True)
                kaggle.api.dataset_download_files(kaggle_dataset, path=download_dir, unzip=True)
                for f in os.listdir(download_dir):
                    if f.endswith('.csv') and 'transformed' not in f:
                        kaggle_file = os.path.join(download_dir, f)
                        logger.info(f"Successfully downloaded and loaded Kaggle file: {kaggle_file}")
                        df = pd.read_csv(kaggle_file)
                        break
            except Exception as kaggle_err:
                logger.warning(f"Kaggle download fallback skipped or failed: {kaggle_err}")

        # Optional MySQL extraction fallback if available
        if df is None or df.empty:
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
            raise CustomETLException("Failed to find and load raw credit card dataset file.", sys)

        logger.info(f"Successfully extracted {len(df)} records with columns: {list(df.columns)}")

        # 2. TRANSFORM
        logger.info("Step 2: Transforming data and engineering debt/repayment risk features...")

        # Standardize column headers
        df.columns = [c.strip().upper().replace('.', '_') for c in df.columns]

        # Customer Identifier
        if 'ID' in df.columns:
            df.rename(columns={'ID': 'CUSTOMER_ID'}, inplace=True)
        elif 'CUSTOMER_ID' not in df.columns:
            df['CUSTOMER_ID'] = range(1, len(df) + 1)

        # Standardize Target Column
        target_col = 'DEFAULT_PAYMENT_NEXT_MONTH'
        if target_col not in df.columns:
            possible_targets = [c for c in df.columns if 'DEFAULT' in c]
            if possible_targets:
                df.rename(columns={possible_targets[0]: target_col}, inplace=True)

        # Categorical Remapping
        if 'EDUCATION' in df.columns:
            logger.info("Remapping invalid EDUCATION codes (0, 5, 6 -> 4 'others')...")
            df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)

        if 'MARRIAGE' in df.columns:
            logger.info("Remapping invalid MARRIAGE codes (0 -> 3 'others')...")
            df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)

        # Feature Engineering: Repayment Delays
        pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6] if f'PAY_{i}' in df.columns]
        if pay_cols:
            logger.info("Engineering repayment delay aggregations (AVG_PAY_DELAY, MAX_PAY_DELAY, NUM_MONTHS_DELAYED)...")
            df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
            df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
            df['NUM_MONTHS_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)

        # Feature Engineering: Bill Statements & Payments
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
        pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]

        if bill_cols:
            logger.info("Engineering bill aggregations & credit utilization metrics...")
            df['TOTAL_BILL_AMT'] = df[bill_cols].sum(axis=1)
            df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
            if 'LIMIT_BAL' in df.columns and 'BILL_AMT1' in df.columns:
                df['CREDIT_UTILIZATION'] = df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1e-5)
            if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
                df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']

        if pay_amt_cols:
            logger.info("Engineering payment aggregations & pay-to-bill ratios...")
            df['TOTAL_PAY_AMT'] = df[pay_amt_cols].sum(axis=1)
            df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)

        if 'TOTAL_PAY_AMT' in df.columns and 'TOTAL_BILL_AMT' in df.columns:
            df['PAY_TO_BILL_RATIO'] = df['TOTAL_PAY_AMT'] / (df['TOTAL_BILL_AMT'] + 1e-5)

        if 'AGE' in df.columns:
            logger.info("Binning AGE into demographic AGE_GROUP categories...")
            bins = [20, 30, 40, 50, 60, 120]
            labels = [0, 1, 2, 3, 4]
            df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, include_lowest=True).astype(int)

        # Outlier clipping via IQR bounds
        logger.info("Applying IQR capping on continuous numeric metrics...")
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

        # Impute missing values if any
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            logger.info(f"Imputing {nan_count} missing values...")
            df.bfill(inplace=True)
            df.ffill(inplace=True)

        logger.info(f"Transformation complete. Dataframe dimensions: {df.shape}")

        # 3. LOAD (Save processed dataset locally)
        logger.info("Step 3: Loading transformed dataset into storage...")
        output_dir = os.path.join(os.path.dirname(__file__), 'include')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'transformed_credit_card_default.csv')

        df.to_csv(output_path, index=False)
        logger.info(f"Saved transformed dataset to: {output_path}")
        logger.info("======================================================================")
        logger.info(f">>> CCDP ETL Pipeline Completed Successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
        logger.info("======================================================================")
        print(f"\nCCDP ETL Pipeline Execution Successful! Transformed {df.shape[0]} rows and {df.shape[1]} columns.")
        print(f"Artifact saved to: {output_path}")
        return df

    except Exception as e:
        logger.error(f"ETL Execution encountered an error: {str(e)}")
        raise CustomETLException(e, sys)


if __name__ == "__main__":
    run_standalone_etl()
