import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from src.CCDP.exception import CustomException
from src.CCDP.logger import logging
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        import pymysql
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established")
        df = pd.read_sql_query('Select * from credit', mydb)
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException(ex, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param.get(model_name, {})

            logging.info(f"Training model: {model_name}")
            if para:
                gs = GridSearchCV(model, para, cv=3, scoring='roc_auc', n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_test_proba = model.predict(X_test)

            test_model_score = roc_auc_score(y_test, y_test_proba)
            report[model_name] = test_model_score
            logging.info(f"Model: {model_name}, ROC-AUC: {test_model_score:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            try:
                return pickle.load(file_obj)
            except Exception:
                import dill
                file_obj.seek(0)
                return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)