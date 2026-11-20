import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.CCDP.exception import CustomException
from src.CCDP.logger import logging
from src.CCDP.utils import save_object


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Feature Engineering (custom domain features)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Engineers 17 behavioral risk features from raw credit card data.

    Features created:
      Delinquency  : AVG_PAY_DELAY, MAX_PAY_DELAY, NUM_MONTHS_DELAYED,
                     RECENCY_WEIGHTED_DELAY, IS_CONSECUTIVE_DELAY
      Bill / Util  : TOTAL_BILL_AMT, AVG_BILL_AMT, CREDIT_UTILIZATION_RECENT,
                     AVG_CREDIT_UTILIZATION, BILL_TREND
      Payment      : TOTAL_PAY_AMT, AVG_PAY_AMT, PAY_TREND
      Debt         : NET_UNPAID_BALANCE, PAY_TO_BILL_RATIO, RECENT_PAY_FULFILLMENT
      Demographics : IS_YOUTH

    Drops redundant older-month bill/payment columns (retained by aggregates).
    Applies IQR-based outlier clipping (fit on train, applied on test/inference).
    """

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
        if 'EDUCATION' in df.columns:
            df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
        if 'MARRIAGE' in df.columns:
            df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
        return df

    def _add_features(self, df):
        df = df.copy()

        if 'PAY_1' in df.columns and 'PAY_0' not in df.columns:
            df.rename(columns={'PAY_1': 'PAY_0'}, inplace=True)

        pay_cols     = [c for c in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'] if c in df.columns]
        bill_cols    = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in df.columns]
        pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in df.columns]

        # 1. Delinquency dynamics
        if pay_cols:
            df['AVG_PAY_DELAY']    = df[pay_cols].mean(axis=1)
            df['MAX_PAY_DELAY']    = df[pay_cols].max(axis=1)
            df['NUM_MONTHS_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)
            weights = np.linspace(len(pay_cols), 1, len(pay_cols), dtype=float)
            df['RECENCY_WEIGHTED_DELAY'] = (df[pay_cols] * weights).sum(axis=1) / weights.sum()
            if 'PAY_0' in df.columns and 'PAY_2' in df.columns:
                df['IS_CONSECUTIVE_DELAY'] = ((df['PAY_0'] > 0) & (df['PAY_2'] > 0)).astype(int)

        # 2. Bill aggregations & utilization
        if bill_cols:
            df['TOTAL_BILL_AMT'] = df[bill_cols].sum(axis=1)
            df['AVG_BILL_AMT']   = df[bill_cols].mean(axis=1)
            if 'LIMIT_BAL' in df.columns:
                df['CREDIT_UTILIZATION_RECENT'] = np.clip(
                    df['BILL_AMT1'] / (df['LIMIT_BAL'] + 1.0), -2.0, 5.0)
                df['AVG_CREDIT_UTILIZATION'] = np.clip(
                    (df['TOTAL_BILL_AMT'] / float(len(bill_cols))) / (df['LIMIT_BAL'] + 1.0), -2.0, 5.0)
            if 'BILL_AMT1' in df.columns and 'BILL_AMT6' in df.columns:
                df['BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']

        # 3. Payment aggregations
        if pay_amt_cols:
            df['TOTAL_PAY_AMT'] = df[pay_amt_cols].sum(axis=1)
            df['AVG_PAY_AMT']   = df[pay_amt_cols].mean(axis=1)
            if 'PAY_AMT1' in df.columns and 'PAY_AMT6' in df.columns:
                df['PAY_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']

        # 4. Debt & payment adequacy
        if 'TOTAL_BILL_AMT' in df.columns and 'TOTAL_PAY_AMT' in df.columns:
            df['NET_UNPAID_BALANCE'] = df['TOTAL_BILL_AMT'] - df['TOTAL_PAY_AMT']
            pos_bill_mask = df['TOTAL_BILL_AMT'] > 0
            df['PAY_TO_BILL_RATIO'] = np.where(
                pos_bill_mask,
                np.clip(df['TOTAL_PAY_AMT'] / (df['TOTAL_BILL_AMT'] + 1.0), 0.0, 5.0),
                1.0
            )
        if 'PAY_AMT1' in df.columns and 'BILL_AMT2' in df.columns:
            pos_bill2_mask = df['BILL_AMT2'] > 0
            df['RECENT_PAY_FULFILLMENT'] = np.where(
                pos_bill2_mask,
                np.clip(df['PAY_AMT1'] / (df['BILL_AMT2'] + 1.0), 0.0, 5.0),
                1.0
            )

        # 5. Demographics
        if 'AGE' in df.columns:
            df['IS_YOUTH'] = (df['AGE'] < 30).astype(int)

        # 6. Drop redundant / multi-collinear columns
        cols_to_drop = [
            'CUSTOMER_ID', 'ID',
            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        return df

    def fit(self, X, y=None):
        try:
            df = X.copy()
            df = self._clean_categoricals(df)
            df = self._add_features(df)
            self.expected_columns = df.columns.tolist()
            for col in self.columns_to_cap:
                if col in df.columns:
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    self.cap_values[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            logging.info(f"FeatureEngineering: produced {len(self.expected_columns)} features")
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            df = X.copy()
            df = self._clean_categoricals(df)
            df = self._add_features(df)
            for col, (lower, upper) in self.cap_values.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=lower, upper=upper)
            if self.expected_columns is not None:
                for col in self.expected_columns:
                    if col not in df.columns:
                        df[col] = 0.0
                df = df[self.expected_columns]
            return df
        except Exception as e:
            raise CustomException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Correlation-based Feature Selection
# ─────────────────────────────────────────────────────────────────────────────
class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    Drop one feature from each pair with absolute Pearson correlation > threshold.

    Why this matters:
      • TOTAL_BILL_AMT vs AVG_BILL_AMT are perfectly correlated (ratio 1:6) — one is redundant.
      • TOTAL_PAY_AMT vs AVG_PAY_AMT — same.
      • AVG_PAY_DELAY and RECENCY_WEIGHTED_DELAY are highly correlated (both avg delay measures).
      Keeping both inflates variance in Logistic Regression, distorts KNN distances,
      and hurts SVM convergence.

    Strategy:
      For each correlated pair (i, j), drop the feature with the higher mean absolute
      correlation across all features (i.e., the one that is most redundant overall).
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        try:
            X = np.array(X, dtype=np.float64)
            corr = np.abs(np.corrcoef(X.T))
            n = corr.shape[0]
            to_drop = set()

            for i in range(n):
                if i in to_drop:
                    continue
                for j in range(i + 1, n):
                    if j in to_drop:
                        continue
                    if corr[i, j] > self.threshold:
                        # Drop the feature with higher mean correlation (more redundant)
                        active = [k for k in range(n) if k not in to_drop]
                        mean_i = np.mean([corr[i, k] for k in active if k != i])
                        mean_j = np.mean([corr[j, k] for k in active if k != j])
                        to_drop.add(j if mean_j >= mean_i else i)

            self.cols_to_keep_ = [i for i in range(n) if i not in to_drop]
            n_dropped = n - len(self.cols_to_keep_)
            logging.info(
                f"CorrelationSelector (threshold={self.threshold}): "
                f"{n} → {len(self.cols_to_keep_)} features  "
                f"(dropped {n_dropped} redundant)"
            )
            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            return np.array(X)[:, self.cols_to_keep_]
        except Exception as e:
            raise CustomException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Mutual Information Feature Selection (SelectKBest)
# ─────────────────────────────────────────────────────────────────────────────
class MutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    Selects top-k features ranked by mutual information with the target.

    Why mutual information over ANOVA F-test (f_classif):
      • Mutual information captures non-linear relationships — critical for
        features like PAY_TO_BILL_RATIO and RECENCY_WEIGHTED_DELAY that have
        non-linear relationships with default probability.
      • f_classif assumes linear association; MI makes no such assumption.

    Why after scaling (not before):
      • MI is scale-invariant, but RobustScaler preceding it ensures numerical
        stability in the kNN-based MI estimator (sklearn uses kNN internally).

    k=20 rationale:
      • After FeatureEngineering (~34 features) and CorrelationSelector (~28 features),
        selecting top 20 by MI removes noise features (demographics, weak aggregates)
        while keeping the dominant signals (PAY delays, utilization, bill ratios).
    """

    def __init__(self, k=20, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            k_actual = min(self.k, X.shape[1])
            self.selector_ = SelectKBest(
                mutual_info_classif, k=k_actual
            )
            self.selector_.fit(X, y)

            self.selected_indices_ = self.selector_.get_support(indices=True)
            scores = self.selector_.scores_
            logging.info(
                f"MutualInfoSelector: {X.shape[1]} → {k_actual} features selected"
            )
            # Log top features by MI score
            ranked = sorted(enumerate(scores), key=lambda x: -x[1])
            logging.info("Top 10 features by Mutual Information score:")
            for rank, (idx, score) in enumerate(ranked[:10], 1):
                logging.info(f"  {rank:2d}. Feature[{idx:2d}]  MI={score:.4f}")

            return self
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, X):
        try:
            return self.selector_.transform(np.array(X, dtype=np.float64))
        except Exception as e:
            raise CustomException(e, sys)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        """
        Full preprocessing pipeline:

          ┌─────────────────────┐
          │  FeatureEngineering │  → 17 new features, IQR clipping, cat cleaning
          │    (~34 features)   │
          └────────┬────────────┘
                   │
          ┌────────▼────────────┐
          │   SimpleImputer     │  → median fill for any remaining NaN
          │   (strategy=median) │
          └────────┬────────────┘
                   │
          ┌────────▼────────────┐
          │  CorrelationSelector│  → drop one of each pair with |corr| > 0.90
          │    (~28 features)   │    (removes TOTAL vs AVG redundancy etc.)
          └────────┬────────────┘
                   │
          ┌────────▼────────────┐
          │    RobustScaler     │  → scale using median/IQR (outlier-robust)
          └────────┬────────────┘
                   │
          ┌────────▼────────────┐
          │  MutualInfoSelector │  → keep top 20 by mutual information with target
          │    (20 features)    │    (non-linear feature ranking, needs y in fit)
          └─────────────────────┘
        """
        try:
            pipeline = Pipeline(steps=[
                ('feature_eng',    FeatureEngineering()),
                ('imputer',        SimpleImputer(strategy='median')),
                ('corr_selector',  CorrelationSelector(threshold=0.90)),
                ('scaler',         RobustScaler()),
                ('mi_selector',    MutualInfoSelector(k=20, random_state=42)),
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info('Read train and test datasets')

            # Standardize column names
            train_df.columns = [c.strip().upper().replace('.', '_') for c in train_df.columns]
            test_df.columns  = [c.strip().upper().replace('.', '_') for c in test_df.columns]

            # Identify target column
            target_column_name = 'DEFAULT_PAYMENT_NEXT_MONTH'
            if target_column_name not in train_df.columns:
                for col in train_df.columns:
                    if 'DEFAULT' in col:
                        target_column_name = col
                        break
                else:
                    target_column_name = train_df.columns[-1]

            # Drop identifiers
            for col in ['ID', 'CUSTOMER_ID']:
                if col in train_df.columns:
                    train_df.drop(columns=[col], inplace=True)
                if col in test_df.columns:
                    test_df.drop(columns=[col], inplace=True)

            logging.info(f"Target column: {target_column_name}")

            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name]
            X_test  = test_df.drop(columns=[target_column_name])
            y_test  = test_df[target_column_name]

            # Log class distribution
            class_counts = y_train.value_counts()
            logging.info(f"Training class distribution: {class_counts.to_dict()}")
            imbalance_ratio = class_counts.max() / class_counts.min()
            logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")

            preprocessing_obj = self.get_data_transformation_object()

            logging.info("Fitting and transforming training data...")
            # Pass y_train so MutualInfoSelector can compute MI scores
            X_train_arr = preprocessing_obj.fit_transform(X_train, y_train)

            logging.info("Transforming test data (no fit on test)...")
            X_test_arr = preprocessing_obj.transform(X_test)

            logging.info(
                f"Final feature shape — Train: {X_train_arr.shape} | Test: {X_test_arr.shape}"
            )

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr  = np.c_[X_test_arr,  np.array(y_test)]

            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
                exist_ok=True
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor saved to artifacts/preprocessor.pkl")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
