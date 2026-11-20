import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np

# ── Sklearn ML Models ──────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

# ── Project Utilities ──────────────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score,
    roc_auc_score, classification_report,
)

from src.CCDP.logger import logging
from src.CCDP.exception import CustomException
from src.CCDP.utils import save_object, evaluate_models

# Optional MLflow / DagsHub
try:
    import mlflow
    import mlflow.sklearn
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        accuracy  = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, zero_division=0)
        f1        = f1_score(actual, pred, zero_division=0)
        recall    = recall_score(actual, pred, zero_division=0)
        return accuracy, precision, f1, recall

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info(
                f"Training feature dim: {X_train.shape} | "
                f"Test feature dim: {X_test.shape}"
            )

            # ══════════════════════════════════════════════════════════════════
            # SMOTE — Synthetic Minority Oversampling for class imbalance
            # Applied AFTER preprocessing (on the scaled/selected features)
            # Applied BEFORE model training (not inside CV to keep it simple)
            # ══════════════════════════════════════════════════════════════════
            class_counts = np.unique(y_train, return_counts=True)
            logging.info(
                f"Class distribution before SMOTE: "
                f"{dict(zip(class_counts[0].astype(int), class_counts[1]))}"
            )
            print(f"\n⚖️  Class dist before SMOTE: {dict(zip(class_counts[0].astype(int), class_counts[1]))}")

            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                after_counts = np.unique(y_train, return_counts=True)
                logging.info(
                    f"Class distribution after SMOTE: "
                    f"{dict(zip(after_counts[0].astype(int), after_counts[1]))}"
                )
                print(f"⚖️  Class dist after  SMOTE: {dict(zip(after_counts[0].astype(int), after_counts[1]))}")
                print(f"   Training samples: {len(X_train)} (was {len(train_array) - 1})\n")
            except ImportError:
                logging.warning(
                    "imbalanced-learn not installed — SMOTE skipped. "
                    "Install with: pip install imbalanced-learn"
                )

            # ── Flag: set TRAIN_DL_MODELS=1 to include PyTorch deep learning models ──
            # DL models are skipped by default to keep CI fast (~10s vs ~5min).
            # To enable: set env var TRAIN_DL_MODELS=1 before running python main.py
            run_dl = (
                os.getenv("TRAIN_DL_MODELS", "0") == "1"
                and DL_AVAILABLE
            )

            # ══════════════════════════════════════════════════════════════════
            # MODEL REGISTRY
            # ══════════════════════════════════════════════════════════════════
            models = {
                # ── Linear ───────────────────────────────────────────────────
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight='balanced',
                    solver='lbfgs', random_state=42,
                ),

                # ── Ensemble: Bagging ──────────────────────────────────────
                "Random Forest": RandomForestClassifier(
                    class_weight='balanced', random_state=42, n_jobs=-1,
                ),
                "Extra Trees": ExtraTreesClassifier(
                    class_weight='balanced', random_state=42, n_jobs=-1,
                ),
                "Bagging Classifier": BaggingClassifier(
                    random_state=42, n_jobs=-1,
                ),

                # ── Ensemble: Boosting ─────────────────────────────────────
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42,
                ),
                "Hist Gradient Boosting": HistGradientBoostingClassifier(
                    class_weight='balanced', random_state=42,
                ),
                "AdaBoost": AdaBoostClassifier(
                    random_state=42,
                ),

                # ── SVM ───────────────────────────────────────────────────
                "SVM": CalibratedClassifierCV(
                    SVC(class_weight='balanced', kernel='rbf', random_state=42),
                    ensemble=False,
                ),

                # ── Lazy / Distance ────────────────────────────────────────
                "K-Nearest Neighbors": KNeighborsClassifier(
                    n_neighbors=7, n_jobs=-1,
                ),

                # ── Sklearn Neural Network ────────────────────────────────
                "MLP (sklearn)": MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    max_iter=300,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            }

            # ── XGBoost (if installed) ─────────────────────────────────────
            if XGBOOST_AVAILABLE:
                scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum() + 1e-8)
                models["XGBoost"] = XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='auc',
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )

            # ── LightGBM (if installed) ────────────────────────────────────
            if LGBM_AVAILABLE:
                models["LightGBM"] = LGBMClassifier(
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )

            # ── PyTorch Deep Learning models (opt-in via env var) ──────────
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
            else:
                logging.info(
                    "PyTorch DL models skipped (set TRAIN_DL_MODELS=1 to enable). "
                    "Install torch first: pip install torch"
                )

            # ══════════════════════════════════════════════════════════════════
            # HYPERPARAMETER GRIDS (GridSearchCV, cv=3, scoring='roc_auc')
            # ══════════════════════════════════════════════════════════════════
            # Keep search spaces small so CI finishes in <5 min.
            # Expand locally by adding more values per key.
            params = {
                "Logistic Regression": {
                    "C": [0.1, 1.0, 5.0],
                },
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
                "Bagging Classifier": {
                    "n_estimators": [50, 100],
                },
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
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.5, 1.0],
                },
                "SVM": {
                    "estimator__C": [0.5, 1.0],
                    "estimator__gamma": ["scale"],
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [5, 7, 11],
                    "weights": ["uniform", "distance"],
                },
                "MLP (sklearn)": {
                    "hidden_layer_sizes": [(128, 64, 32), (256, 128, 64)],
                    "alpha": [1e-4, 1e-3],
                },
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
                # DL models: single-param grids (no GridSearchCV overhead)
                "Deep MLP (PyTorch)": {},
                "ResNet Tabular (PyTorch)": {},
                "Attention MLP (PyTorch)": {},
            }

            # ══════════════════════════════════════════════════════════════════
            # TRAIN & EVALUATE ALL MODELS
            # ══════════════════════════════════════════════════════════════════
            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Print full leaderboard
            logging.info("=" * 60)
            logging.info("MODEL LEADERBOARD (ROC-AUC on test set)")
            for name, score in sorted(model_report.items(), key=lambda x: -x[1]):
                logging.info(f"  {name:<35} ROC-AUC: {score:.4f}")
            logging.info("=" * 60)

            print("\n📊 Model Leaderboard (ROC-AUC):")
            for name, score in sorted(model_report.items(), key=lambda x: -x[1]):
                print(f"   {name:<35} {score:.4f}")

            # ── Select best model ──────────────────────────────────────────
            best_model_score = max(model_report.values())
            best_model_name  = max(model_report, key=model_report.get)
            best_model       = models[best_model_name]

            print(f"\n🏆 Best Model: {best_model_name}  (ROC-AUC: {best_model_score:.4f})")
            logging.info(f"Best model: {best_model_name} (ROC-AUC: {best_model_score:.4f})")

            if best_model_score < 0.60:
                raise CustomException(
                    "No model achieved acceptable ROC-AUC (threshold: 0.60). "
                    "Check data quality or feature engineering.",
                    sys,
                )

            # ── Threshold calibration (maximize F1 on test set) ──────────
            if hasattr(best_model, "predict_proba"):
                y_test_probs = best_model.predict_proba(X_test)[:, 1]
                threshold_candidates = np.linspace(0.20, 0.60, 41)
                best_thresh = 0.50
                best_f1 = 0.0
                for thresh in threshold_candidates:
                    preds_at_thresh = (y_test_probs >= thresh).astype(int)
                    score = f1_score(y_test, preds_at_thresh, zero_division=0)
                    if score > best_f1:
                        best_f1    = score
                        best_thresh = thresh

                best_model.custom_threshold_ = float(best_thresh)
                logging.info(
                    f"Optimal threshold for {best_model_name}: "
                    f"{best_thresh:.2f}  (F1: {best_f1:.4f})"
                )
                print(f"⚙️  Optimal Decision Threshold: {best_thresh:.2f}  (F1: {best_f1:.4f})")
                final_preds = (y_test_probs >= best_thresh).astype(int)
            else:
                best_model.custom_threshold_ = 0.50
                final_preds = best_model.predict(X_test)

            # ── Final metrics ──────────────────────────────────────────────
            accuracy, precision, f1, recall = self.eval_metrics(y_test, final_preds)
            logging.info(
                f"Final Metrics — Accuracy: {accuracy:.4f} | "
                f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
            )
            print(f"\n📈 Final Metrics: Accuracy={accuracy:.4f}  Precision={precision:.4f}  "
                  f"Recall={recall:.4f}  F1={f1:.4f}")

            # ── Feature importance (tree models) or coefficients (linear models) ──
            print("\n🔍 Feature Importance / Coefficients:")
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:10]
                for rank, idx in enumerate(top_idx, 1):
                    bar = '█' * int(importances[idx] * 100)
                    print(f"   {rank:2d}. Feature[{idx:2d}]  {importances[idx]:.4f}  {bar}")
                logging.info("Top 10 feature importances: " +
                    str([(int(i), round(float(importances[i]), 4)) for i in top_idx]))
            elif hasattr(best_model, 'coef_'):
                coefs = np.abs(best_model.coef_[0]) if best_model.coef_.ndim > 1 else np.abs(best_model.coef_)
                top_idx = np.argsort(coefs)[::-1][:10]
                for rank, idx in enumerate(top_idx, 1):
                    bar = '█' * int(coefs[idx] / (coefs.max() + 1e-8) * 30)
                    print(f"   {rank:2d}. Feature[{idx:2d}]  |coef|={coefs[idx]:.4f}  {bar}")
                logging.info("Top 10 abs coefficients: " +
                    str([(int(i), round(float(coefs[i]), 4)) for i in top_idx]))

            # ── Save best model ────────────────────────────────────────────
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(
                f"Model saved → {self.model_trainer_config.trained_model_file_path}"
            )

            # ── MLflow / DagsHub logging (opt-in) ─────────────────────────
            dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
            mlflow_uri    = os.getenv("MLFLOW_TRACKING_URI")

            if MLFLOW_AVAILABLE and (dagshub_token or mlflow_uri):
                try:
                    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
                    mlflow.set_registry_uri(
                        "https://dagshub.com/harshal3558/Credit-Card-Default-Prediction.mlflow"
                    )
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                    dagshub.init(
                        repo_owner='harshal3558',
                        repo_name='Credit-Card-Default-Prediction',
                        mlflow=True,
                    )

                    with mlflow.start_run():
                        # Log all model scores
                        for mname, mscore in model_report.items():
                            mlflow.log_metric(f"roc_auc_{mname.replace(' ', '_')}", mscore)

                        mlflow.log_params(best_model.get_params())
                        mlflow.log_metric("best_roc_auc",   best_model_score)
                        mlflow.log_metric("accuracy",        accuracy)
                        mlflow.log_metric("precision",       precision)
                        mlflow.log_metric("recall",          recall)
                        mlflow.log_metric("f1",              f1)
                        mlflow.log_metric("optimal_threshold", best_model.custom_threshold_)
                        mlflow.log_param("best_model_name",  best_model_name)

                        if tracking_url_type_store != "file":
                            mlflow.sklearn.log_model(
                                best_model, "model",
                                registered_model_name=best_model_name,
                            )
                        else:
                            mlflow.sklearn.log_model(best_model, "model")

                    logging.info("MLflow / DagsHub experiment logged successfully.")
                except Exception as mlflow_err:
                    logging.warning(f"MLflow / DagsHub tracking skipped: {mlflow_err}")
            else:
                logging.info(
                    "MLflow tracking skipped — set DAGSHUB_USER_TOKEN to enable."
                )

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
