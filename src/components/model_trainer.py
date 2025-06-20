import os
import sys
import dill
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomeException
from src.logger import logger
from src.utils import save_obj, evaluate_model

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 
from sklearn.model_selection import RandomizedSearchCV 
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

@dataclass
class ModelTrainerConfig:
    model_trainer_obj_file_path = os.path.join("artifacts", "model_trainer.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_obj_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, preprocess_obj_file_path):
        try:
            logger.info("Spliting the train and test data to feed the model")
            self.input_train_features = np.delete(train_arr, 7, axis=1)
            self.input_train_target = train_arr[:, 7]

            self.input_test_features = np.delete(test_arr, 7, axis=1)
            self.input_test_target = test_arr[:, 7]
            
            logger.info("Creating the models and training the model")
            models = {"Linear Regression" : LinearRegression(),"Ridge": Ridge(),
                     "Lasso": Lasso(),
                     "RandomForestRegressor" : RandomForestRegressor(),
                     "K-Neighboures" : KNeighborsRegressor(),
                     "Decision Tree" : DecisionTreeRegressor(),
                     "XGBRegressor" : XGBRegressor(),
                     "CatBoostRegressor" : CatBoostRegressor(verbose=False),
                     "AdaBoostRegressor" : AdaBoostRegressor()
                    }
           
            best_model = evaluate_model(X_train=self.input_train_features, y_train=self.input_train_target, X_test=self.input_test_features, y_test=self.input_test_target, models=models)
            best_model.fit(self.input_train_features, self.input_train_target)
            
            logger.info("Saving the model Obj")
            save_obj(
                file_path = self.model_trainer_obj_config.model_trainer_obj_file_path,
                obj = best_model
            )



        except Exception as e:
            raise CustomeException(e, sys)

