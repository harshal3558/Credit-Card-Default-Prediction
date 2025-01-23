from src.CCDP.logger import logging
from src.CCDP.exception import CustomException
# from src.CCDP.components import data_ingestion
from src.CCDP.components.data_ingestion import DataIngestion
from src.CCDP.components.data_ingestion import DataIngestionConfig
from src.CCDP.components.data_transformation import DataTransformationConfig
from src.CCDP.components.data_transformation import DataTransformation
import sys

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion=data_ingestion()
        # data_ingestion_config = DataIngestionConfig()
        
        data_ingestion = DataIngestion()
        # data_ingestion.initiate_data_ingestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        # data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path,test_data_path)


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)