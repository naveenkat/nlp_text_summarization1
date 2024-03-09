import os,sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from logger import logging
from exception import CustomException
from components import data_ingestion
from components import data_transformation
from components import model_trainer

class training_config:
    def __init__(self):
        pass
    def initiate_training(self):
        data_ingestion_object = data_ingestion.DataIngestion()
        self.train_data,self.test_data,self.valid_data,self.data_ingestion_path = data_ingestion_object.initiate_data_ingestion()
    def initiate_transformation(self):
        data_transformation_object = data_transformation.DataTransformation(self.test_data,self.valid_data)
        self.test_transform,self.valid_transform = data_transformation_object.initiate_data_transformation()
    def initiate_model_trainer1(self):
        model_trainer_object = model_trainer.ModelTrainer(self.test_transform,self.valid_transform)
        model_trainer_object.initiate_model_trainer()



if __name__ == "__main__":
    training_config_object = training_config()
    training_config_object.initiate_training()
    training_config_object.initiate_transformation()
    training_config_object.initiate_model_trainer1()
    

