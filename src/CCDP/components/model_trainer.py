import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np

# Importing classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, classification_report
)

from src.CCDP.logger import logging
from src.CCDP.exception import CustomException
from src.CCDP.utils import save_object, evaluate_models

# Optional MLflow and DagsHub tracking
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
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)
        recall = recall_score(actual, pred, zero_division=0)
        return accuracy, precision, f1, recall

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info(f"Training feature dimension: {X_train.shape}, Test feature dimension: {X_test.shape}")

            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
                'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
            params = {
                "Logistic Regression": {
                    'C': [1.0]
                },
                "Random Forest": {
                    'n_estimators': [150],
                    'max_depth': [10],
                    'min_samples_leaf': [10]
                },
                "Gradient Boosting": {
                    'n_estimators': [100],
                    'learning_rate': [0.08],
                    'max_depth': [4]
                }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            # Get best model score and name based on ROC-AUC
            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            best_model = models[best_model_name]

            print(f"Best model selected: {best_model_name} (ROC-AUC: {best_model_score:.4f})")
            logging.info(f"Best model selected: {best_model_name} with ROC-AUC: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable ROC-AUC (threshold: 0.60)", sys)

            # Optimize decision threshold for class imbalance (default detection)
            if hasattr(best_model, "predict_proba"):
                y_test_probs = best_model.predict_proba(X_test)[:, 1]
                threshold_candidates = np.linspace(0.20, 0.60, 41)
                best_thresh = 0.50
                best_f1 = 0.0
                for thresh in threshold_candidates:
                    preds_at_thresh = (y_test_probs >= thresh).astype(int)
                    score = f1_score(y_test, preds_at_thresh, zero_division=0)
                    if score > best_f1:
                        best_f1 = score
                        best_thresh = thresh

                best_model.custom_threshold_ = float(best_thresh)
                logging.info(f"Calibrated decision threshold for {best_model_name}: {best_thresh:.2f} (Test F1: {best_f1:.4f})")
                print(f"Optimal Decision Threshold: {best_thresh:.2f} (F1-score: {best_f1:.4f})")
                final_preds = (y_test_probs >= best_thresh).astype(int)
            else:
                best_model.custom_threshold_ = 0.50
                final_preds = best_model.predict(X_test)

            accuracy, precision, f1, recall = self.eval_metrics(y_test, final_preds)
            logging.info(f"Final Model Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Save the trained best model locally
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Model saved locally at: {self.model_trainer_config.trained_model_file_path}")

            # MLflow and DagsHub logging (with safe fallback)
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.set_registry_uri("https://dagshub.com/harshal3558/Credit-Card-Default-Prediction.mlflow")
                    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                    dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
                    if dagshub_token or os.getenv("MLFLOW_TRACKING_URI"):
                        dagshub.init(repo_owner='harshal3558', repo_name='Credit-Card-Default-Prediction', mlflow=True)
                    else:
                        logging.info("Running local MLflow tracking without remote hang.")

                    with mlflow.start_run():
                        mlflow.log_params(best_model.get_params())
                        mlflow.log_metric("roc_auc", best_model_score)
                        mlflow.log_metric("accuracy", accuracy)
                        mlflow.log_metric("precision", precision)
                        mlflow.log_metric("recall", recall)
                        mlflow.log_metric("f1", f1)
                        mlflow.log_metric("optimal_threshold", best_model.custom_threshold_)

                        if tracking_url_type_store != "file":
                            mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model_name)
                        else:
                            mlflow.sklearn.log_model(best_model, "model")
                except Exception as mlflow_err:
                    logging.warning(f"MLflow / DagsHub tracking skipped due to: {mlflow_err}")

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)