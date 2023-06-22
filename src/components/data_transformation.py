# handle missing value
#outlier treatment
# handle imbalance dataset

import os, sys
import pandas as pd
from src.logger import logging
import numpy as np
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join("artifact/data_transformation", "preprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def get_data_transformation_obj(self):
        try:
            logging.info(" Data Transformation Started")

            numerical_features =  ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer" , SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                    
                    
                ]
            )
            
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ])

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def remote_outliers_IQR(self,col,df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            
            iqr = Q3-Q1
            
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            
            df.loc[(df[col]>upper_limit), col] = upper_limit
            df.loc[(df[col]<lower_limit), col] = lower_limit
            
            return df
            
            
        except Exception as e:
            logging.info('outlier handling code')
            raise CustomException(e, sys)
        
        
    def inititate_data_transformation(self,train_path, test_path):
        
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            
            for col in numerical_features:
                self.remote_outliers_IQR(col = col, df = train_data)
            
            logging.info('outlier capped on our train data')
            
            for col in numerical_features:
                self.remote_outliers_IQR(col = col, df = test_data)
            
            logging.info('outlier capped on our test data')
              
            preprocess_obj = self.get_data_transformation_obj()
            
            target_columns = "label"
            drop_columns = [target_columns]
            
            logging.info('splitting train data into dependent & independent features')
            input_feature_train_data = train_data.drop(drop_columns, axis = 1)
            target_feature_train_data = train_data[target_columns]
            
            logging.info('splitting test data into dependent & independent features')
            input_feature_test_data = test_data.drop(drop_columns, axis = 1)
            target_feature_test_data = test_data[target_columns]
            
            # apply transformation on our train dataq and test data
            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            
            input_test_arr = preprocess_obj.transform(input_feature_test_data)
            
            # apply preprocessor object on our train data and test data  
            
            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]  
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]  
              
            save_object(file_path=self.data_transformation_config.preprocess_obj_file_path,
                        obj=preprocess_obj)
            
            return(train_array,
                   test_array,
                   self.data_transformation_config.preprocess_obj_file_path)
              
        except Exception as e:
            logging.info('outlier handling code')
            raise CustomException(e, sys)
        