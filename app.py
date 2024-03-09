from fastapi import FastAPI
import uvicorn
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.pipelines import predict_pipeline
from src.logger import logging
from src.exception import CustomException

text:str = "what is text summarization?"
app = FastAPI()
@app.get("/",tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")
@app.post("/predict")
async def predict_route(text):
    try:
        obj = predict_pipeline.PredictPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        logging.info("error occured in app.py")
        raise CustomException(e,sys)


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port = 8080)