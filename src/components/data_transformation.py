import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomeException
from src.logger import logger
from src.utils import save_obj

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import pickle
import dill

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
       self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        logger.info("The transformation has started")
        try:
            cat_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                            'test preparation course']
            num_features = [ 'reading score','writing score']

            cat_pipeline = Pipeline(
                steps=[ 
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )

            
            logger.info(f"Categorical columns: {num_features}")
            logger.info(f"Numerical columns: {cat_features}")

            preprocessor = ColumnTransformer(
                [
                    ("Cat transform", cat_pipeline, cat_features),
                    ("Num transform", num_pipeline, num_features)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomeException(e, sys)

    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_set_df = pd.read_csv(train_path)
            test_set_df = pd.read_csv(test_path)

            logger.info("reading the test, train data complete")
            logger.info("Getting the preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            numerical_columns = ['reading score', 'writing score']
            target_columns = 'math score'

            input_train_df = train_set_df.drop(target_columns, axis=1)  #X_train
            target_train_df = train_set_df[target_columns]  #y_train

            input_test_df = test_set_df.drop(target_columns, axis=1) #X_test
            target_test_df = test_set_df[target_columns] #y_test

            logger.info("applying the preprocessing obj on train and test df")

            input_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[
                    input_train_arr, np.array(target_train_df)
                ]
            test_arr = np.c_[input_test_arr, np.array(target_test_df)]

            logger.info(f"Saved preprocessing object.")

            #saving the preprocessing obj
            save_obj(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomeException(e, sys)


