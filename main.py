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
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        ## Model Training
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
