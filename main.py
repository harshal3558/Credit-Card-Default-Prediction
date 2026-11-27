import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"  # Suppress MLflow stderr hint

from src.CCDP.logger import logging
from src.CCDP.exception import CustomException
from src.CCDP.components.data_ingestion import DataIngestion
from src.CCDP.components.data_ingestion import DataIngestionConfig
from src.CCDP.components.data_transformation import DataTransformation
from src.CCDP.components.data_transformation import DataTransformationConfig
from src.CCDP.components.model_trainer import ModelTrainerConfig
from src.CCDP.components.model_trainer import ModelTrainer
from src.CCDP.components.model_monitering import ModelMonitor
from src.CCDP.utils import load_object

import sys
import io

# Fix for UnicodeEncodeError in Windows console when libraries print emojis
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        ## Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

        ## Production Drift Monitoring Baseline Check
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_trainer.model_trainer_config.trained_model_file_path)
        feature_names = getattr(preprocessor, "selected_feature_names_", None)
        y_prob_ref = model.predict_proba(train_arr[:, :-1])[:, 1] if hasattr(model, "predict_proba") else None
        y_prob_cur = model.predict_proba(test_arr[:, :-1])[:, 1] if hasattr(model, "predict_proba") else None

        monitor = ModelMonitor()
        drift_report = monitor.evaluate_drift(
            train_arr[:, :-1],
            test_arr[:, :-1],
            feature_names=feature_names,
            y_prob_ref=y_prob_ref,
            y_prob_cur=y_prob_cur,
        )
        print(f"Drift Monitoring Status: {drift_report['overall_status']} (Critical: {len(drift_report['critical_features'])}, Warning: {len(drift_report['warning_features'])})")

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
