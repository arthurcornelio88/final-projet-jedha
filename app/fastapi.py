from fastapi import FastAPI
from app.data_validation import PredictionInput
from app.utils import (
    load_pipeline_from_mlflow,
    get_latest_run_id
)
import pandas as pd
import mlflow.pyfunc

# Initialize FastAPI application
app = FastAPI()

# Model and pipeline configurations
MODEL_NAME = "decision_tree"
MODEL_VERSION = "6"  # Use the latest version
RUN_ID = get_latest_run_id("hyperparameter_tuning")
PIPELINE_ARTIFACT_PATH = "pipeline/preprocessing_pipeline.pkl"  # Path inside the MLflow artifacts

# Load model and pipeline dynamically from MLflow
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
pipeline = load_pipeline_from_mlflow(RUN_ID, PIPELINE_ARTIFACT_PATH)

# Endpoint for predictions
@app.post("/predictions")
def get_predictions(input: PredictionInput):
    """
    Endpoint to get predictions based on input data.
    """
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([input.model_dump()])  # Wrap in a list to create a DataFrame

    # Apply the pipeline to preprocess the data

    prepared_data = pipeline.transform(input_data)
    print("Preprocessing input done!")

    # Perform predictions
    prediction = model.predict(prepared_data)

    # Format the predictions to two decimal places
    formatted_prediction = [round(p, 2) for p in prediction]

    return {f"real value: {input_data["price"]}, prediction: {formatted_prediction}"}
