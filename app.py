from src.CCDP.logger import logging
from src.CCDP.exception import CustomException
from src.CCDP.components.data_ingestion import DataIngestion
from src.CCDP.components.data_ingestion import DataIngestionConfig
from src.CCDP.components.data_transformation import DataTransformation
from src.CCDP.components.data_transformation import DataTransformationConfig
from src.CCDP.components.model_tranier import ModelTrainerConfig
from src.CCDP.components.model_tranier import ModelTrainer

import sys

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion() 
        # data_ingestion.initiate_data_ingestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        # data_transformation_config=DataIngestionConfig()
        data_transformation=DataTransformation()
        # data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
        ## Model Training

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)