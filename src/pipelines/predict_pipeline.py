import os,sys  
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.logger import logging
from src.exception import CustomException
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,Text2TextGenerationPipeline
from transformers import pipeline

class PredictPipeline:
    

    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(r"C:\text_summarization_project\notebooks\tokenizer")
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
        model_path = AutoModelForSeq2SeqLM.from_pretrained(r"C:\text_summarization_project\notebooks\pegasus-samsum-model")

        pipe = pipeline("summarization", model=model_path,tokenizer=tokenizer)

        print("Dialogue:")
        print(text)

        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)

        return output
