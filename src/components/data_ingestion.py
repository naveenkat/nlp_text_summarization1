from transformers import pipeline, set_seed
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset, load_metric
from dataclasses import dataclass
import os,sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from exception import CustomException
from logger import logging


from tqdm import tqdm

@dataclass
class DataIngestionConfig:
    train_path = os.path.join('artifacts','train.csv')
    test_path = os.path.join('artifacts','test.csv')
    validation_path = os.path.join('artifacts','validation.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    
    try:
        logging.info('data ingestion step is started')
        def initiate_data_ingestion(self):
            data_ingestion_file_path = os.makedirs(os.path.dirname(self.data_ingestion_config.train_path),exist_ok=True)
            dataset_samsum = load_dataset("samsum")
            train_data = dataset_samsum["train"]
            test_data =  dataset_samsum["test"]
            valid_data = dataset_samsum["validation"]
            train_data.to_csv(os.path.join(self.data_ingestion_config.train_path))
            test_data.to_csv(os.path.join(self.data_ingestion_config.test_path))
            valid_data.to_csv(os.path.join(self.data_ingestion_config.validation_path))
            logging.info("data ingestion step is completed")
            return train_data,test_data,valid_data,data_ingestion_file_path
    except Exception as e:
        logging.info("exception occured at data ingestion step")
        raise CustomException(e,sys)
                  




        

    
    