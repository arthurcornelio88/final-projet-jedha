from app.model import load_model
from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data
import numpy as np
import pandas as pd
import glob
import os

def get_last_modified_model(folder_path):
    # Get all files in the folder
    models = glob.glob(os.path.join(folder_path, "*"))
    
    # Return the most recently modified file
    if models:
        last_modified_model = max(models, key=os.path.getmtime)
        return last_modified_model
    else:
        return None

def test_model_predictions():

    datafile = "data/total_data.csv"
    df_raw =  load_data(datafile)
    
    folder_path = "models/"  # Replace with your folder path
    last_model = get_last_modified_model(folder_path)

    if last_model:
        print("Last modified file:", last_model)
    else:
        print("No files found in the folder.")

    model = load_model(last_model)

    X_train, y_train, X_test, y_test, pipeline = process_and_pipeline(df_raw)

    # Test prediction shape
    predictions = model.predict(X_test)

    ### ASSERTS ###

    # Ensure the loaded model is not None or corrupted
    assert model is not None, "Model could not be loaded!"
    
    # Evaluating if number of predictions are equal to number of samples 
    assert predictions.shape[0] == X_test.shape[0], "Number of predictions does not match the number of samples in X_test!"

    # Ensure that there are no NaN or infinite values in the predictions
    assert np.all(np.isfinite(predictions)), "Predictions contain NaN or infinite values!"

    # ensure that predictions fall within a reasonable range (e.g., prices should not be negative)
    assert (predictions >= 0).all(), "Predictions contain negative values!"

    # Basic Sanity Check
    mae = np.mean(np.abs(predictions - y_test))
    assert mae < 1e6, f"Mean Absolute Error is too large: {mae}"

