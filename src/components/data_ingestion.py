import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import customException
from sklearn.model_selection import train_test_split
from src.components.data_transformation import dataTransformationConfig
from src.components.data_transformation import dataTransformation
from src.components.model_trainer import modelTrainerConfig
from src.components.model_trainer import modelTrainer


#USE DATA CLASS WHEN YOU ONLY WAN TO INITIALIZE VARIABLE IF IT ALSO HAS FUNCTION GO WITH NORMAL CLASSES
@dataclass
class dataIngestionConfig:
    train_data_path:str =os.path.join('artifacts',"train.csv")
    test_data_path:str =os.path.join('artifacts',"test.csv")
    raw_data_path:str =os.path.join('artifacts',"raw.csv")
    
    
class dataIngestion:
    def __init__(self):
        self.dataIngestion=dataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entering data ingestion method")
            
        try:
            df=pd.read_csv("notebook/data/stud.csv")
            logging.info("READ THE DATA SET AS DATA FRAME")
            
            #below line returns parental directory 
            os.makedirs(os.path.dirname(self.dataIngestion.train_data_path),exist_ok=True)
            
            #storing raw 
            df.to_csv(self.dataIngestion.raw_data_path,index=False,header=True)
            logging.info("INITIATE TRAIN AND TEST SET")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.dataIngestion.train_data_path,index=False,header=True)
            test_set.to_csv(self.dataIngestion.test_data_path,index=False,header=True)
            
            logging.info("DATA INGESTION COMPLETED")
            
            #returning as it will be beneficial in pipelining
            return(
                self.dataIngestion.train_data_path,
                self.dataIngestion.test_data_path
            )
        except Exception as e:
            
            raise customException(e,sys)
        
        
if __name__ == "__main__":
    obj=dataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
            
    data_transformation=dataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_tranformation(train_data,test_data)    
    
    modelTrainer=modelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))