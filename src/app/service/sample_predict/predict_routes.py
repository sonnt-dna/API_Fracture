
from fastapi import Header, APIRouter, HTTPException, FastAPI, UploadFile, File, Request, Form, Query
from pydantic import BaseModel
import pandas as pd
import json as json
import csv
from io import StringIO
from .predict_handlers import predict_fracture
from typing import List


# Create new "APIRouter" object
predict_router = APIRouter()

# Check if the server is running or not
@predict_router.get("/hello")
async def read_main():
    return {"message": "Hello World", "Note": "This is a test"}

# Endpoint to predict
@predict_router.post("/predict", status_code=201)
async def upload_csv(csv_file: UploadFile = File(...),
                    feature: str = None,
                    scoring: str = "R2",
                    objective: str = "valid_score",
                    algorithm: str = "xgboost",
                    iteration: int = None,
                    target: str = None,
                    ):
    # Check if the uploaded file is a CSV file
    if csv_file.content_type != "text/csv":
        raise HTTPException(status_code=415, detail="File attached is not a CSV file")

    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file.file)
    except:
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    finally:
        csv_file.file.close()
    feature_list = feature.split(",") if feature else []
    # target_list = target.split(",") if target else []
    predicted_result = predict_fracture(df, feature_list, scoring, objective, algorithm, iteration, target)

    return predicted_result