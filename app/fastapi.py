from fastapi import FastAPI
from app.model import load_model
from app.pipeline import process_and_pipeline
from app.static_values import TEST_INPUT
from app.data_validation import PredictionInput
import pandas as pd
import json
import joblib

# Initialize FastAPI application
app = FastAPI()

# Define paths to model and pipeline
MODEL_PATH = "models/20250113_00-36-27 - trained_model.pkl"  # Adjust the path
PIPELINE_PATH = "models/20250113_00-36-27 - preprocessing_pipeline.pkl"  # Adjust the path for the saved pipeline
MODEL_PLACEMENT = "local"  # "mlflow" or "local"

# Load the model
model = load_model(model_placement=MODEL_PLACEMENT, model_path=MODEL_PATH)

# Load the pipeline
pipeline = joblib.load(PIPELINE_PATH)

# Endpoint to get predictions
@app.post("/predictions")
def get_predictions(input: PredictionInput):
    """
    Endpoint to get predictions based on input data.
    """
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([input.model_dump()])  # Wrap in a list to create a DataFrame

    # Apply the pipeline to preprocess the data
    prepared_data = pipeline.transform(input_data)

    # Perform predictions
    prediction = model.predict(prepared_data)

    return {"prediction": prediction.tolist()}


# Endpoint to test with static values
@app.get("/test-prediction")
def test_prediction():
    """
    Endpoint to test the model with static values.
    """
    json_filepath = "data/json/10-20_rows.json"

    # Open the file and load the JSON content
    with open(json_filepath, "r") as f:
        test_data = json.load(f)  # Load the content as a list of dictionaries

    # Convert the test data to a DataFrame
    test_df = pd.DataFrame(test_data)

    # Apply the pipeline to preprocess the data
    prepared_data = pipeline.transform(test_df[0:2])

    # Perform predictions
    prediction = model.predict(prepared_data)

    return {"test_input": test_data, "prediction": prediction.tolist()}
