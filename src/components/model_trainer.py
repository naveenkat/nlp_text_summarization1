from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import pandas as pd
from datasets import load_dataset, load_metric
from dataclasses import dataclass
from utils.my_utils import save_object
import os,sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from exception import CustomException
from logger import logging


@dataclass
class ModelTrainerConfig:
        preprocessor_path = os.path.join("artifacts","pegasus-model")
        model_path = os.path.join("artifacts","tokenizer")

class ModelTrainer:
        def __init__(self,test_data_transform,valid_data_transform):
                self.model_trainer_config = ModelTrainerConfig()
                self.test_data_transform = test_data_transform
                self.valid_data_transform = valid_data_transform
                self.model_ckpt = "google/pegasus-cnn_dailymail"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
                self.model =  AutoModelForSeq2SeqLM.from_pretrained(self.model_ckpt)


        def get_trainer_object(self):
                trainer_args = TrainingArguments(
                output_dir='pegasus-samsum', num_train_epochs=1, warmup_steps=500,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                weight_decay=0.01, logging_steps=10,
                evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
                gradient_accumulation_steps=16
                                               )
                
                seq2seq_data_collator = DataCollatorForSeq2Seq(self.tokenizer,model = self.model)
                 
                trainer = Trainer(model=self.model, args=trainer_args,
                  tokenizer=self.tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=self.test_data_transform,
                  eval_dataset=self.valid_data_transform)
                return trainer
                
        def initiate_model_trainer(self):
                model_obj =self.get_trainer_object()

                model_output_dir = "artifacts/pegasus-samsum-model"
                tokenizer_output_dir = "artifacts/pegasus-samsum-tokenizer"

                
                model_obj.train()
                model_obj.model.save_pretrained(model_output_dir)
                self.tokenizer.save_pretrained(tokenizer_output_dir)





                
                
                
                

                