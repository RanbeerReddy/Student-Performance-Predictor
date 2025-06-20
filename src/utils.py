import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logger

from src.exception import CustomeException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise Exception(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        def model_evaluation(y_test, pred):
            mse = mean_squared_error(y_test,pred)
            r2_scoree = r2_score(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            return mse, r2_scoree, mae
        
        y_train_metrics = pd.DataFrame({"Name": [], "MSE": [], "r2_score": [], "MAE":[], "Estimator":[]})
        y_test_metrics = pd.DataFrame({"Name": [], "MSE": [], "r2_score": [], "MAE":[], "Estimator":[]})

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
    
            y_train_mse, y_train_r2, y_train_mse = model_evaluation(y_train_pred, y_train) 
            y_test_mse, y_test_r2, y_test_mae = model_evaluation(y_test_pred, y_test) 

            y_train_metrics.loc[i] = [list(models.keys())[i], y_train_mse, y_train_r2, y_train_mse, list(models.values())[i]]
            y_test_metrics.loc[i] = [list(models.keys())[i], y_test_mse, y_test_r2, y_test_mae, list(models.values())[i]]
        
        #Selecting the model with best r2 score
        best_model_index = y_test_metrics.loc[y_test_metrics['r2_score'].idxmax()]
        best_model = best_model_index['Estimator']

        return best_model
            
        
    except Exception as e:
        raise CustomeException(e, sys)