import sys
import pandas as pd
from src.exception import CustomeException 
from src.utils import load_obj
import os

class predictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            self.preprocess_file_path = os.path.join("artifacts", "preprocessor.pkl")
            self.model_file_path = os.path.join("artifacts", "model_trainer.pkl")
            
            self.preprocess_obj = load_obj(self.preprocess_file_path)
            self.model_obj = load_obj(self.model_file_path)

            features = self.preprocess_obj.transform(features)
            self.predicted_output = self.model_obj.predict(features)

            return self.predicted_output 
        except Exception as e:
            raise CustomeException(e, sys)

class CustomData:
    def __init__(self,
        gender: str, 
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race/ethnicity" : [self.race_ethnicity],
                "parental level of education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test preparation course" : [self.test_preparation_course],
                "reading score" : [self.reading_score],
                "writing score" : [self.writing_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomeException(e, sys)
        