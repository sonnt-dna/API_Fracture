
from fastapi import Header, APIRouter, HTTPException, FastAPI, UploadFile, File, Request, Form, Query
from pydantic import BaseModel
import pandas as pd
import json as json
import csv
from io import StringIO
from .predict_handlers import predict, input_csv_schema, input_params_schema

# Create new "APIRouter" object
predict_router = APIRouter()

# Check if the server is running or not
@predict_router.get("/hello")
async def read_main():
    return {"message": "Hello World", "Note": "This is a test"}

# Endpoint to predict
@predict_router.post("/predict", status_code=201)
async def upload_csv(csv_file: UploadFile = File(...), query_params: str = Form(...)):

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


    # Validate query_params using QueryParams schema
    try:
        # query_params_in_json = json.dumps(query_params)
        parameter = json.loads(query_params)
        input_params_schema(**parameter)
    except ValueError as e:
        return {"detail": str(e)}
    except Exception as e:
        return {"detail": "Invalid JSON"}

    predicted_result = predict(df, parameter)

    return predicted_result