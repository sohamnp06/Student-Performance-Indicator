import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import customException
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class dataTransformation:
    def __init__(self):
        self.data_transformation_config=dataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        THIS FUNCTION IS RESPONSIBLE FOR DATA TRANSFORMATION
        '''
        try:
            num_features=['reading_score', 'writing_score']
            cat_features=['gender',
                        'race_ethnicity',
                        'parental_level_of_education',
                        'lunch',
                        'test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                    
                ]
            )
            
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("ohe",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("NUMERICAL COLUMNS AND CAATEGORCIAL COLUMSNN ENCODING ")
        
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
                
            )
            
            return preprocessor
        
        except Exception as e:
            raise customException(e,sys)
        
        
        
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("READ TRAIN AND TEST DATA")
            
            logging.info("obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math_score"
            numerical_column_name=['reading_score', 'writing_score']
            
            input_featue_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_featue_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            
            input_featue_train_arr=preprocessing_obj.fit_transform(input_featue_train_df)
            input_featue_test_arr=preprocessing_obj.transform(input_featue_test_df)
            
            train_arr=np.c_[
                input_featue_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_featue_test_arr,np.array(target_feature_test_df)
            ]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
               raise customException (e,sys)
            
            
            
            