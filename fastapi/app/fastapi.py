from fastapi import FastAPI
from app.data_validation import PredictionInput
from app.data_preprocessing import clean_price
from app.utils import (
    load_pipeline_from_mlflow,
    get_latest_run_id
)
import pandas as pd
import mlflow.pyfunc
import os

# Initialize FastAPI application
app = FastAPI()

# Load configurations from environment variables (injected via ConfigMap)
MODEL_NAME = os.getenv("MODEL_NAME", "decision_tree")  # Default: "decision_tree"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")  # Default: "latest"
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "hyperparameter_tuning")  # Default experiment name
ARTIFACT_PATH = os.getenv("PIPELINE_ARTIFACT_PATH", "pipeline/preprocessing_pipeline.pkl")  # Default artifact path

# MLflow tracking server URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_BACKEND_STORE_URI", "sqlite:///mlflow.db")  # Default local SQLite
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the latest run ID
RUN_ID = get_latest_run_id(EXPERIMENT_NAME)

# Load model and preprocessing pipeline from MLflow
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}")
pipeline = load_pipeline_from_mlflow(RUN_ID, ARTIFACT_PATH)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

# Endpoint for predictions
@app.post("/predictions")
def get_predictions(input: PredictionInput):
    """
    Endpoint to get predictions based on input data.
    """
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([input.model_dump()])  # Wrap in a list to create a DataFrame

        # Apply the pipeline to preprocess the data

        prepared_data = pipeline.transform(input_data)
        print("Preprocessing input done!")

        # Perform predictions
        prediction = model.predict(prepared_data)

        # Apply cleaning to the price column
        input_data["price"] = input_data["price"].apply(clean_price)

        # Format the real prices to two decimal places
        formatted_real_price = [round(float(r), 2) for r in input_data["price"]]

        # Format the predictions to two decimal places
        formatted_prediction = [round(p, 2) for p in prediction]

        # Return a properly structured dictionary with model details
        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,  # Include model version
            "real_values": formatted_real_price,
            "predictions": formatted_prediction
        }
    except Exception as e:
        return {"error": str(e)}
