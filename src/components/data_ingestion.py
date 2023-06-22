import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.components.model_trainer import modelTrainer

from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation 

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifact/data_ingestion',"train.csv")
    test_data_path = os.path.join('artifact/data_ingestion',"test.csv")
    raw_data_path = os.path.join('artifact/data_ingestion',"raw.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("data ingestion started")
        
    def inititate_data_ingestion(self):
        try:
            logging.info("data reading using pandas library from local system")
            data = pd.read_csv(os.path.join("Crop_recommendation.csv"))
            logging.info("data reading completed")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("data splited into train test")
            
            train_set,test_set =train_test_split(data,test_size =.20, random_state =42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False , header = True)
            
            logging.info("data ingestion completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.info("error occured in data ingestion")
            raise CustomException(e,sys)
    
    
   
   
if __name__ =="__main__": 
    obj = DataIngestion()
    train_data_path , test_data_path = obj.inititate_data_ingestion()
    
    
    data_transformation = DataTransformation()
    
    train_arr, test_arr, _ = data_transformation.inititate_data_transformation(train_data_path , test_data_path)
    
    
    modeltrainer = modelTrainer()
    print(modeltrainer.inititate_model_trainer(train_arr, test_arr))
    
    
# src\components\data_ingestion.py