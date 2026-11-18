import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from src.CCDP.exception import CustomException
from src.CCDP.logger import logging
from src.CCDP.utils import save_object


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_cap = [
            'LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2',
            'TOTAL_BILL_AMT', 'AVG_BILL_AMT', 'TOTAL_PAY_AMT', 'AVG_PAY_AMT',
            'NET_UNPAID_BALANCE', 'BILL_TREND', 'PAY_TREND',
            'AVG_PAY_DELAY', 'MAX_PAY_DELAY', 'RECENCY_WEIGHTED_DELAY'
        ]
        self.cap_values = {}
        self.expected_columns = None

    def _clean_categoricals(self, df):
        # Education: 0, 5, 6 are undocumented / others -> map to 4 (Others)
        if 'EDUCATION' in df.columns:
            df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
        
        # Marriage: 0 is undocumented -> map to 3 (Others)
        if 'MARRIAGE' in df.columns:
            df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)

        return df

    def _add_features(self, df):
        df = df.copy()

        # Remap column names if PAY_1 was provided instead of PAY_0
        if 'PAY_1' in df.columns and 'PAY_0' not in df.columns:
            df.rename(columns={'PAY_1': 'PAY_0'}, inplace=True)

        pay_cols = [c for c in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] if c in df.columns]
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
        pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]

        # 1. Delinquency & Repayment Delay Dynamics (highest credit default signals)
        if pay_cols:
            df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
            df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
            df['NUM_MONTHS_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)
            
            # Recency-weighted delay (recent months carry higher penalty)
            weights = np.linspace(len(pay_cols), 1, len(pay_cols), dtype=float)
            df['RECENCY_WEIGHTED_DELAY'] = (df[pay_cols] * weights).sum(axis=1) / weights.sum()

            if 'PAY_0' in df.columns and 'PAY_2' in df.columns:
                df['IS_CONSECUTIVE_DELAY'] = ((df['PAY_0'] > 0) & (df['PAY_2'] > 0)).astype(int)

        # 2. Bill Aggregations & Credit Utilization
        if bill_cols:
            df['TOTAL_BILL_AMT'] = df[bill_cols].sum(axis=1)
            df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
            
            if 'LIMIT_BAL' in df.columns:
                # Safe ratio division with reasonable bounds
                df['CREDIT_UTILIZATION_RECENT'] = np.clip(
                    df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1.0), -2.0, 5.0
                )
                df['AVG_CREDIT_UTILIZATION'] = np.clip(
                    (df['TOTAL_BILL_AMT'] / float(len(bill_cols))) / (df['LIMIT_BAL'] + 1.0), -2.0, 5.0
                )
            
            if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
                df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']

        # 3. Payment Aggregations & Trends
        if pay_amt_cols:
            df['TOTAL_PAY_AMT'] = df[pay_amt_cols].sum(axis=1)
            df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)

            if 'PAY_AMT1' in df.columns and 'PAY_AMT6' in df.columns:
                df['PAY_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']

        # 4. Debt Accumulation & Payment Adequacy
        if 'TOTAL_BILL_AMT' in df.columns and 'TOTAL_PAY_AMT' in df.columns:
            df['NET_UNPAID_BALANCE'] = df['TOTAL_BILL_AMT'] - df['TOTAL_PAY_AMT']

            # Pay to Bill Ratio: safe against zero/negative bills
            positive_bill_mask = df['TOTAL_BILL_AMT'] > 0
            df['PAY_TO_BILL_RATIO'] = np.where(
                positive_bill_mask,
                np.clip(df['TOTAL_PAY_AMT'] / (df['TOTAL_BILL_AMT'] + 1.0), 0.0, 5.0),
                1.0
            )

        if 'PAY_AMT1' in df.columns and 'BILL_AMT2' in df.columns:
            # Payment fulfillment of previous month's statement
            positive_bill2_mask = df['BILL_AMT2'] > 0
            df['RECENT_PAY_FULFILLMENT'] = np.where(
                positive_bill2_mask,
                np.clip(df['PAY_AMT1'] / (df['BILL_AMT2'] + 1.0), 0.0, 5.0),
                1.0
            )

        # 5. Demographics (Keep AGE numeric, add youth risk flag)
        if 'AGE' in df.columns:
            df['IS_YOUTH'] = (df['AGE'] < 30).astype(int)

        # 6. Drop redundant/multi-collinear columns and identifiers
        # Retain BILL_AMT1, BILL_AMT2, PAY_AMT1, PAY_AMT2 as recent direct signals
        cols_to_drop = [
            'CUSTOMER_ID', 'ID',
            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]
        existing_drops = [col for col in cols_to_drop if col in df.columns]
        df.drop(columns=existing_drops, inplace=True)

        return df

    def fit(self, X, y=None):
        try:
            df = X.copy()
            df = self._clean_categoricals(df)
            df = self._add_features(df)

            self.expected_columns = df.columns.tolist()

            # Compute IQR capping bounds on train set
            for col in self.columns_to_cap:
                if col in df.columns:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    self.cap_values[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            df = X.copy()
            df = self._clean_categoricals(df)
            df = self._add_features(df)

            # Apply Clipping using fitted bounds
            for col, (lower, upper) in self.cap_values.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=lower, upper=upper)

            # Ensure consistent column ordering and handle any missing columns
            if self.expected_columns is not None:
                for col in self.expected_columns:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df[self.expected_columns]

            return df
        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            preprocessing_pipeline = Pipeline(steps=[
                ('feature_eng', FeatureEngineering()),
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            return preprocessing_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading the train and test file')

            # Standardize Column Names
            train_df.columns = [c.strip().upper().replace('.', '_') for c in train_df.columns]
            test_df.columns = [c.strip().upper().replace('.', '_') for c in test_df.columns]

            target_column_name = 'DEFAULT_PAYMENT_NEXT_MONTH'
            if target_column_name not in train_df.columns:
                for col in train_df.columns:
                    if 'DEFAULT' in col:
                        target_column_name = col
                        break
                else:
                    target_column_name = train_df.columns[-1]

            # Drop identifier columns ('ID', 'CUSTOMER_ID') if they exist
            for col in ['ID', 'CUSTOMER_ID']:
                if col in train_df.columns:
                    train_df.drop(columns=[col], inplace=True)
                if col in test_df.columns:
                    test_df.drop(columns=[col], inplace=True)

            logging.info(f"Target column identified: {target_column_name}")

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            preprocessing_obj = self.get_data_transformation_object()

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
