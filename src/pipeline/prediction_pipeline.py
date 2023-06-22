# create prediction pipeline class -> completed
# create function for load a object -> completed
# create custom class based upon our dataset -> completed
# create function to convert data into dataframe with the help of dict
import os , sys
from src.logger import logging
from src.exception import CustmeException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object
from src.exception import CustomException


class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self , features):
        preprocessor_path = os.path.join("artifact/data_transformation", "preprocessor.pkl")
        model_path = os.path.join("artifact/model_trainer","model.pkl")
        
        processor = load_object(preprocessor_path)
        model = load_object(model_path)
        
        scaled = processor.transform(features)
        pred = model.predict(scaled)
        
        return pred
    
# numerical_features =  ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

class CustomClass:
    def __init__(self, 
                  N:int,
                  P:int, 
                  K:int, 
                  temperature:int, 
                  humidity:int,
                  ph:int,  
                  rainfall:int,
                  ):
        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall
        
        
        
        
    
    def get_data_DataFrame(self):
        try:
            custom_input = {
                "N": [self.N],
                "P": [self.P],
                "K":[self.K],
                "temperature":[self.temperature],
                "humidity":[self.humidity],
                "ph":[self.ph],
                "rainfall":[self.rainfall],
                
            }

            data= pd.DataFrame(custom_input)
            return data
        
        except Exception as e:
            raise CustmeException(e,sys)

        
        
        