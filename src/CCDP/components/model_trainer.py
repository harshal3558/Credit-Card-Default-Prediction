import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
import numpy as np
import pandas as pd

# ── Sklearn ML Models ──────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

try:
    from sklearn.frozen import FrozenEstimator
    FROZEN_AVAILABLE = True
except ImportError:
    FROZEN_AVAILABLE = False

# ── Boosting Libraries ─────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ── Imbalanced-Learn (Leakage-Free SMOTE inside CV Pipeline) ──────────────────
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# ── Deep Learning Models (PyTorch — optional) ─────────────────────────────────
try:
    from src.CCDP.components.dl_model import (
        DeepMLPClassifier,
        ResNetTabularClassifier,
        AttentionMLPClassifier,
    )
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# ── Metrics & Project Utilities ────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.CCDP.exception import CustomException
from src.CCDP.logger import logging
from src.CCDP.utils import evaluate_models, save_object

# Optional MLflow / DagsHub
try:
    import dagshub
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    governance_report_path: str = os.path.join("artifacts", "governance_report.json")
    # Business cost assumptions (relative units: missed default costs 5x false alarm)
    fn_cost: float = 5.0
    fp_cost: float = 1.0


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred, probs=None):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)
        recall = recall_score(actual, pred, zero_division=0)
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if probs is not None:
            metrics["roc_auc"] = float(roc_auc_score(actual, probs))
            metrics["pr_auc"] = float(average_precision_score(actual, probs))
            metrics["brier_score"] = float(brier_score_loss(actual, probs))
        return metrics

    def _compute_expected_cost(self, y_true, y_pred):
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        total_cost = fn * self.model_trainer_config.fn_cost + fp * self.model_trainer_config.fp_cost
        return total_cost / max(1, len(y_true))

    def _calibrate_model(self, base_model, X_val, y_val):
        """
        Calibrates model probabilities on a held-out validation split using isotonic
        regression, keeping the calibrated wrapper only if it improves Brier score.
        """
        if not hasattr(base_model, "predict_proba"):
            return base_model, None, None

        raw_probs = base_model.predict_proba(X_val)[:, 1]
        brier_before = float(brier_score_loss(y_val, raw_probs))

        try:
            if FROZEN_AVAILABLE:
                calibrated = CalibratedClassifierCV(
                    FrozenEstimator(base_model), method="isotonic"
                )
            else:
                calibrated = CalibratedClassifierCV(
                    base_model, method="isotonic", cv="prefit"
                )
            calibrated.fit(X_val, y_val)
            cal_probs = calibrated.predict_proba(X_val)[:, 1]
            brier_after = float(brier_score_loss(y_val, cal_probs))

            if brier_after <= brier_before:
                logging.info(
                    f"Isotonic calibration improved Brier score: {brier_before:.4f} -> {brier_after:.4f}"
                )
                return calibrated, brier_before, brier_after
            else:
                logging.info(
                    f"Calibration skipped (Brier before={brier_before:.4f}, after={brier_after:.4f})"
                )
                return base_model, brier_before, brier_before
        except Exception as cal_err:
            logging.warning(f"Probability calibration skipped: {cal_err}")
            return base_model, brier_before, brier_before

    def _optimize_threshold(self, y_val, y_val_probs):
        """
        Optimizes decision threshold on the validation split (never on test set)
        using business expected cost (FN=5.0, FP=1.0), breaking ties by F1-score.
        """
        candidates = np.linspace(0.15, 0.65, 51)
        best_thresh = 0.50
        best_cost = float("inf")
        best_f1 = 0.0

        for thresh in candidates:
            preds = (y_val_probs >= thresh).astype(int)
            cost = self._compute_expected_cost(y_val, preds)
            f1 = f1_score(y_val, preds, zero_division=0)
            if cost < best_cost - 1e-6 or (abs(cost - best_cost) <= 1e-6 and f1 > best_f1):
                best_cost = cost
                best_f1 = f1
                best_thresh = float(thresh)

        return best_thresh, best_cost, best_f1

    @staticmethod
    def _compute_fairness_audit(y_true, y_pred, test_csv_path=os.path.join("artifacts", "test.csv")):
        """
        Audits group fairness (Selection Rate, Disparate Impact, TPR, FPR) across
        demographic groups (SEX, AGE < 30) on the test set.
        """
        if not os.path.exists(test_csv_path):
            return {"status": "skipped", "reason": "artifacts/test.csv not found"}

        test_df = pd.read_csv(test_csv_path)
        test_df.columns = [c.strip().upper().replace(".", "_") for c in test_df.columns]
        if len(test_df) != len(y_true):
            return {"status": "skipped", "reason": "row count mismatch with test.csv"}

        audit = {}
        groups = {}
        if "SEX" in test_df.columns:
            groups["SEX"] = test_df["SEX"].map({1: "Male", 2: "Female"}).fillna("Other")
        if "AGE" in test_df.columns:
            groups["AGE_GROUP"] = pd.Series(
                np.where(test_df["AGE"] < 30, "Youth (<30)", "Adult (>=30)"),
                index=test_df.index,
            )

        for attr, series in groups.items():
            attr_metrics = {}
            selection_rates = []
            for grp_val in sorted(pd.Series(series).unique()):
                mask = (np.asarray(series) == grp_val)
                if mask.sum() == 0:
                    continue
                yt = y_true[mask]
                yp = y_pred[mask]
                sel_rate = float(np.mean(yp == 1))
                tpr = float(recall_score(yt, yp, zero_division=0))
                tn = float(np.sum((yt == 0) & (yp == 0)))
                fp = float(np.sum((yt == 0) & (yp == 1)))
                fpr = fp / max(1.0, fp + tn)
                selection_rates.append(sel_rate)
                attr_metrics[str(grp_val)] = {
                    "count": int(mask.sum()),
                    "positive_rate": round(sel_rate, 4),
                    "tpr_recall": round(tpr, 4),
                    "fpr": round(fpr, 4),
                }

            di_ratio = (
                round(min(selection_rates) / max(selection_rates), 4)
                if selection_rates and max(selection_rates) > 0
                else 1.0
            )
            audit[attr] = {
                "disparate_impact_ratio": di_ratio,
                "four_fifths_rule_pass": bool(di_ratio >= 0.80),
                "groups": attr_metrics,
            }

        return audit

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training data into train_sub (80%) and val (20%) for leakage-free calibration & threshold tuning")
            X_full_train, y_full_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            X_train, X_val, y_train, y_val = train_test_split(
                X_full_train,
                y_full_train,
                test_size=0.20,
                stratify=y_full_train,
                random_state=42,
            )

            logging.info(
                f"Shapes — Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}"
            )

            run_dl = os.getenv("TRAIN_DL_MODELS", "0") == "1" and DL_AVAILABLE

            # ══════════════════════════════════════════════════════════════════
            # MODEL REGISTRY (Leakage-Free CV + Balanced Class Handling)
            # ══════════════════════════════════════════════════════════════════
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
                "Extra Trees": ExtraTreesClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1
                ),
                "Bagging Classifier": BaggingClassifier(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Hist Gradient Boosting": HistGradientBoostingClassifier(
                    class_weight="balanced", random_state=42
                ),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Linear SVM": CalibratedClassifierCV(
                    LinearSVC(class_weight="balanced", random_state=42, max_iter=2000),
                    ensemble=False,
                ),
                "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
                "MLP (sklearn)": MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    max_iter=200,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            }

            if XGBOOST_AVAILABLE:
                scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum() + 1e-8)
                models["XGBoost"] = XGBClassifier(
                    eval_metric="auc",
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )

            if LGBM_AVAILABLE:
                models["LightGBM"] = LGBMClassifier(
                    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
                )

            if run_dl:
                logging.info("TRAIN_DL_MODELS=1 detected — including PyTorch DL models.")
                models["Deep MLP (PyTorch)"] = DeepMLPClassifier(
                    hidden_dims=(256, 128, 64, 32), dropout=0.3,
                    lr=1e-3, epochs=60, batch_size=256, random_state=42,
                )
                models["ResNet Tabular (PyTorch)"] = ResNetTabularClassifier(
                    hidden_dim=128, n_blocks=4, dropout=0.25,
                    lr=1e-3, epochs=60, batch_size=256, random_state=42,
                )
                models["Attention MLP (PyTorch)"] = AttentionMLPClassifier(
                    hidden_dims=(128, 64, 32), dropout=0.3,
                    lr=1e-3, epochs=60, batch_size=256, random_state=42,
                )

            params = {
                "Logistic Regression": {"C": [0.1, 1.0, 5.0]},
                "Random Forest": {
                    "n_estimators": [150],
                    "max_depth": [10, 20],
                    "min_samples_leaf": [5, 10],
                },
                "Extra Trees": {
                    "n_estimators": [150],
                    "max_depth": [10, 20],
                    "min_samples_leaf": [5],
                },
                "Bagging Classifier": {"n_estimators": [50, 100]},
                "Gradient Boosting": {
                    "n_estimators": [100],
                    "learning_rate": [0.08, 0.1],
                    "max_depth": [4],
                },
                "Hist Gradient Boosting": {
                    "max_iter": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [4, 6],
                },
                "AdaBoost": {"n_estimators": [50, 100], "learning_rate": [0.5, 1.0]},
                "Linear SVM": {"estimator__C": [0.1, 1.0]},
                "K-Nearest Neighbors": {"n_neighbors": [5, 11], "weights": ["distance"]},
                "MLP (sklearn)": {"hidden_layer_sizes": [(128, 64, 32)], "alpha": [1e-4, 1e-3]},
                "XGBoost": {
                    "n_estimators": [100],
                    "learning_rate": [0.08],
                    "max_depth": [4, 6],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                },
                "LightGBM": {
                    "n_estimators": [100],
                    "learning_rate": [0.08],
                    "num_leaves": [31, 63],
                    "min_child_samples": [20],
                },
                "Deep MLP (PyTorch)": {},
                "ResNet Tabular (PyTorch)": {},
                "Attention MLP (PyTorch)": {},
            }

            # ── Train & evaluate on validation split (test set remains untouched!) ──
            model_report: dict = evaluate_models(X_train, y_train, X_val, y_val, models, params)

            print("\n📊 Model Leaderboard (Validation ROC-AUC):")
            for name, score in sorted(model_report.items(), key=lambda x: -x[1]):
                print(f"   {name:<35} {score:.4f}")
                logging.info(f"  {name:<35} Val ROC-AUC: {score:.4f}")

            best_val_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            print(f"\n🏆 Best Model: {best_model_name} (Val ROC-AUC: {best_val_score:.4f})")
            if best_val_score < 0.60:
                raise CustomException("No model achieved acceptable ROC-AUC (threshold: 0.60).", sys)

            # ── Probability Calibration (on Validation Split) ─────────────────
            calibrated_model, brier_before, brier_after = self._calibrate_model(
                best_model, X_val, y_val
            )

            # ── Business-Cost Threshold Optimization (on Validation Split) ────
            if hasattr(calibrated_model, "predict_proba"):
                y_val_probs = calibrated_model.predict_proba(X_val)[:, 1]
                best_thresh, val_cost, val_f1 = self._optimize_threshold(y_val, y_val_probs)
            else:
                best_thresh, val_cost, val_f1 = 0.50, 0.0, 0.0

            calibrated_model.custom_threshold_ = float(best_thresh)
            calibrated_model.base_estimator_name_ = best_model_name
            print(f"⚙️  Optimal Decision Threshold (Val Cost-Opt): {best_thresh:.2f} (Val F1: {val_f1:.4f})")

            # ── Final Untouched Test-Set Evaluation ───────────────────────────
            if hasattr(calibrated_model, "predict_proba"):
                y_test_probs = calibrated_model.predict_proba(X_test)[:, 1]
                final_preds = (y_test_probs >= best_thresh).astype(int)
                default_preds = (y_test_probs >= 0.50).astype(int)
                test_metrics = self.eval_metrics(y_test, final_preds, y_test_probs)
                cost_opt = self._compute_expected_cost(y_test, final_preds)
                cost_default = self._compute_expected_cost(y_test, default_preds)
            else:
                y_test_probs = None
                final_preds = calibrated_model.predict(X_test)
                test_metrics = self.eval_metrics(y_test, final_preds)
                cost_opt = self._compute_expected_cost(y_test, final_preds)
                cost_default = cost_opt

            cost_savings_pct = (
                ((cost_default - cost_opt) / cost_default * 100.0) if cost_default > 0 else 0.0
            )

            print(
                f"\n📈 Untouched Test Metrics: "
                f"ROC-AUC={test_metrics.get('roc_auc', 0):.4f}  "
                f"PR-AUC={test_metrics.get('pr_auc', 0):.4f}  "
                f"Brier={test_metrics.get('brier_score', 0):.4f}  "
                f"F1={test_metrics['f1']:.4f}  "
                f"Cost Savings vs 0.50={cost_savings_pct:.1f}%"
            )

            # ── Fairness & Governance Report ──────────────────────────────────
            fairness_audit = self._compute_fairness_audit(y_test, final_preds)
            governance_report = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "best_model": best_model_name,
                "validation_leaderboard_roc_auc": {
                    k: round(float(v), 4)
                    for k, v in sorted(model_report.items(), key=lambda x: -x[1])
                },
                "calibration": {
                    "brier_before": round(brier_before, 5) if brier_before is not None else None,
                    "brier_after": round(brier_after, 5) if brier_after is not None else None,
                },
                "business_cost_optimization": {
                    "fn_cost": self.model_trainer_config.fn_cost,
                    "fp_cost": self.model_trainer_config.fp_cost,
                    "optimal_threshold": round(best_thresh, 3),
                    "test_expected_cost_at_optimal": round(cost_opt, 4),
                    "test_expected_cost_at_0_50": round(cost_default, 4),
                    "cost_reduction_pct": round(cost_savings_pct, 2),
                },
                "untouched_test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
                "fairness_audit": fairness_audit,
            }

            os.makedirs(os.path.dirname(self.model_trainer_config.governance_report_path), exist_ok=True)
            with open(self.model_trainer_config.governance_report_path, "w", encoding="utf-8") as f:
                json.dump(governance_report, f, indent=2)
            logging.info(f"Governance report saved → {self.model_trainer_config.governance_report_path}")

            # ── Save Best Calibrated Model ────────────────────────────────────
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=calibrated_model,
            )
            logging.info(f"Model saved → {self.model_trainer_config.trained_model_file_path}")

            # ── MLflow / DagsHub logging (opt-in) ─────────────────────────────
            dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

            if MLFLOW_AVAILABLE and (dagshub_token or mlflow_uri):
                try:
                    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
                    mlflow.set_registry_uri(
                        "https://dagshub.com/harshal3558/Credit-Card-Default-Prediction.mlflow"
                    )
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    dagshub.init(
                        repo_owner="harshal3558",
                        repo_name="Credit-Card-Default-Prediction",
                        mlflow=True,
                    )

                    with mlflow.start_run():
                        for mname, mscore in model_report.items():
                            mlflow.log_metric(f"val_roc_auc_{mname.replace(' ', '_')}", mscore)
                        for mkey, mval in test_metrics.items():
                            mlflow.log_metric(f"test_{mkey}", mval)
                        mlflow.log_metric("optimal_threshold", best_thresh)
                        mlflow.log_metric("cost_reduction_pct", cost_savings_pct)
                        mlflow.log_param("best_model_name", best_model_name)

                        if tracking_url_type_store != "file":
                            mlflow.sklearn.log_model(
                                calibrated_model, "model", registered_model_name=best_model_name
                            )
                        else:
                            mlflow.sklearn.log_model(calibrated_model, "model")

                    logging.info("MLflow / DagsHub experiment logged successfully.")
                except Exception as mlflow_err:
                    logging.warning(f"MLflow / DagsHub tracking skipped: {mlflow_err}")
            else:
                logging.info("MLflow tracking skipped — set DAGSHUB_USER_TOKEN to enable.")

            return test_metrics.get("roc_auc", test_metrics["accuracy"])

        except Exception as e:
            raise CustomException(e, sys)
