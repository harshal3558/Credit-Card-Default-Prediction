import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.CCDP.exception import CustomException
from src.CCDP.logger import logging
from src.CCDP.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            # df=pd.read_csv(r"C:\Users\asd\Desktop\Credit Card Default Prediction\artifacts\raw.csv")

            # numerical_columns=[col for col in df.columns if df[col].dtype != "O"]
            # logging.info('Numerical columns are:',numerical_columns)
            numerical_columns=['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            # categorical_columns=[col for col in df.columns if df[col].dtype == "O"]
            # logging.info('Categorical Columns are:',categorical_columns)

            num_pipeline=Pipeline(steps=[
                #Handling missing values
                ('imputer',SimpleImputer(strategy='median')),
                #Implement Standardisation
                ('scaler',StandardScaler())
            ])
            
            # cat_pipeline=Pipeline(steps=[
            #     ('imputer',SimpleImputer(strategy='most_frequent')),
            #     ('One_hot_encoder',OneHotEncoder()),
            #     ('scalar',StandardScaler(with_mean=False))
            # ])

            # logging.info(f"Categorical:{categorical_columns}")
            logging.info(f"Numerical Columns:{numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    # ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Reading the train and test file')

            preprocessing_obj=self.get_data_transformation_object()

            target_column_name='default.payment.next.month'
            numerical_column=['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

            ## divide the train dataset to idependent and dependent feature

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            ## divide the test dataset to independent and dependent feature

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )




        except Exception as e:
            raise CustomException(e,sys)