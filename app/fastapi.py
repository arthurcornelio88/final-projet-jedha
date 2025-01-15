from fastapi import FastAPI
from app.data_validation import PredictionInput
from app.data_preprocessing import clean_price
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
MODEL_VERSION = "9"  # Use the latest version
RUN_ID = get_latest_run_id("hyperparameter_tuning")
PIPELINE_ARTIFACT_PATH = "pipeline/preprocessing_pipeline.pkl"  # Path inside the MLflow artifacts

# Load model and pipeline dynamically from MLflow
# model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/latest") # Alternatively, load the latest version (use "latest")
pipeline = load_pipeline_from_mlflow(RUN_ID, PIPELINE_ARTIFACT_PATH)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

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
