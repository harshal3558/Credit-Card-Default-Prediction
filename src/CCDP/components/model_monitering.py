"""
model_monitering.py — Production Data & Model Drift Monitoring.

Computes statistical stability metrics between a reference (training) distribution
and a current (test / production) distribution:
  1. Population Stability Index (PSI) per feature and for predicted probabilities
  2. Two-sample Kolmogorov-Smirnov (KS) test per feature and for predicted scores
  3. Automated alert classification:
       - OK       : PSI < 0.10
       - WARNING  : 0.10 <= PSI < 0.25
       - CRITICAL : PSI >= 0.25 (or KS p-value < 0.001 with PSI >= 0.10)

Outputs a structured JSON report at `artifacts/drift_report.json`.
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import ks_2samp

from src.CCDP.exception import CustomException
from src.CCDP.logger import logging


@dataclass
class ModelMonitorConfig:
    drift_report_path: str = os.path.join("artifacts", "drift_report.json")
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    n_bins: int = 10


class ModelMonitor:
    """Monitors feature and prediction score drift using PSI and KS tests."""

    def __init__(self, config: Optional[ModelMonitorConfig] = None):
        self.config = config or ModelMonitorConfig()

    @staticmethod
    def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) between reference and current 1D arrays.
        Uses quantile-based bin edges from the reference distribution.
        """
        ref = np.asarray(reference, dtype=np.float64)
        cur = np.asarray(current, dtype=np.float64)
        ref = ref[~np.isnan(ref)]
        cur = cur[~np.isnan(cur)]

        if len(ref) == 0 or len(cur) == 0 or np.std(ref) == 0:
            return 0.0

        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.unique(np.percentile(ref, quantiles))
        if len(bin_edges) < 2:
            return 0.0

        # Extend outer edges to cover unseen min/max in current data
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref, bins=bin_edges)
        cur_counts, _ = np.histogram(cur, bins=bin_edges)

        eps = 1e-6
        ref_pct = np.maximum(ref_counts / len(ref), eps)
        cur_pct = np.maximum(cur_counts / len(cur), eps)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(0.0, psi)

    def _classify_psi(self, psi_value: float) -> str:
        if psi_value >= self.config.psi_critical_threshold:
            return "CRITICAL"
        if psi_value >= self.config.psi_warning_threshold:
            return "WARNING"
        return "OK"

    def evaluate_drift(
        self,
        X_ref: np.ndarray,
        X_cur: np.ndarray,
        feature_names: Optional[List[str]] = None,
        y_prob_ref: Optional[np.ndarray] = None,
        y_prob_cur: Optional[np.ndarray] = None,
        save_report: bool = True,
    ) -> Dict:
        """
        Run feature-level and prediction-level drift checks and optionally save JSON report.
        """
        try:
            X_ref = np.asarray(X_ref, dtype=np.float64)
            X_cur = np.asarray(X_cur, dtype=np.float64)
            n_features = X_ref.shape[1]

            if feature_names is None or len(feature_names) != n_features:
                feature_names = [f"Feature_{i}" for i in range(n_features)]

            feature_reports = {}
            warning_features = []
            critical_features = []

            for idx, name in enumerate(feature_names):
                ref_col = X_ref[:, idx]
                cur_col = X_cur[:, idx]

                psi = self.compute_psi(ref_col, cur_col, n_bins=self.config.n_bins)
                ks_stat, ks_pval = ks_2samp(ref_col, cur_col)
                status = self._classify_psi(psi)

                if status == "CRITICAL":
                    critical_features.append(name)
                elif status == "WARNING":
                    warning_features.append(name)

                feature_reports[name] = {
                    "psi": round(float(psi), 5),
                    "ks_statistic": round(float(ks_stat), 5),
                    "ks_p_value": round(float(ks_pval), 5),
                    "ref_mean": round(float(np.mean(ref_col)), 4),
                    "cur_mean": round(float(np.mean(cur_col)), 4),
                    "status": status,
                }

            prediction_drift = None
            if y_prob_ref is not None and y_prob_cur is not None:
                pred_psi = self.compute_psi(y_prob_ref, y_prob_cur, n_bins=self.config.n_bins)
                pred_ks, pred_pval = ks_2samp(y_prob_ref, y_prob_cur)
                prediction_drift = {
                    "psi": round(float(pred_psi), 5),
                    "ks_statistic": round(float(pred_ks), 5),
                    "ks_p_value": round(float(pred_pval), 5),
                    "ref_mean_prob": round(float(np.mean(y_prob_ref)), 4),
                    "cur_mean_prob": round(float(np.mean(y_prob_cur)), 4),
                    "status": self._classify_psi(pred_psi),
                }

            overall_status = "OK"
            if critical_features or (prediction_drift and prediction_drift["status"] == "CRITICAL"):
                overall_status = "CRITICAL"
            elif warning_features or (prediction_drift and prediction_drift["status"] == "WARNING"):
                overall_status = "WARNING"

            report = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "overall_status": overall_status,
                "reference_samples": int(len(X_ref)),
                "current_samples": int(len(X_cur)),
                "n_features_monitored": int(n_features),
                "critical_features": critical_features,
                "warning_features": warning_features,
                "prediction_score_drift": prediction_drift,
                "features": feature_reports,
            }

            if save_report:
                os.makedirs(os.path.dirname(self.config.drift_report_path), exist_ok=True)
                with open(self.config.drift_report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                logging.info(f"Drift report saved to {self.config.drift_report_path} (status={overall_status})")

            return report
        except Exception as e:
            raise CustomException(e, sys)
