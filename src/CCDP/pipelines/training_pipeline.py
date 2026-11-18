import os
import sys
import io

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.getcwd())

from src.CCDP.exception import CustomException
from src.CCDP.logger import logging
from src.CCDP.components.data_ingestion import DataIngestion
from src.CCDP.components.data_transformation import DataTransformation
from src.CCDP.components.model_trainer import ModelTrainer

# Fix for UnicodeEncodeError in Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if __name__ == "__main__":
    try:
        logging.info("Starting the training pipeline")

        # 1. Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")

        # 2. Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation completed.")

        # 3. Model Training
        model_trainer = ModelTrainer()
        accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model Training completed. Final Accuracy: {accuracy}")
        print(f"Model Training Pipeline completed successfully. Accuracy: {accuracy}")

    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
