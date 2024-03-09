from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
class DataTransformation:
    try:
        logging.info("data transformation has started")
        def __init__(self,test_data,valid_data):
            self.data_transformation_config = DataTransformationConfig()
            self.test_data = test_data
            self.valid_data = valid_data
            self.model_ckpt = "google/pegasus-cnn_dailymail"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)

        def convert_examples_to_features(self,x):
            input_encodings = self.tokenizer(x['dialogue'] , max_length = 1024, truncation = True )

            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(x['summary'], max_length = 128, truncation = True )
            return {
                        'input_ids' : input_encodings['input_ids'],
                        'attention_mask': input_encodings['attention_mask'],
                        'labels': target_encodings['input_ids']
                   }
        
        def initiate_data_transformation(self):
            test_data_transform = self.test_data.map(self.convert_examples_to_features, batched = True)
            valid_data_transform = self.valid_data.map(self.convert_examples_to_features, batched = True)
            print(test_data_transform[0]["input_ids"])

            logging.info("data transformation step is completed")
            return test_data_transform,valid_data_transform
    except Exception as e:
        logging.info("error occured in data transformation")
        raise CustomException(e,sys)

        

    

    









    
