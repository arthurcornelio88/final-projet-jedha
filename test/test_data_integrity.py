from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data
import pandas as pd

def test_data_processing():

    data_file = 'data/total_data.csv'  # Path to your dataset

    df_raw = load_data(data_file)
    print("Data loaded.")

    X_train, y_train, X_test, y_test, pipeline = process_and_pipeline(df_raw)
    print("Data processed.")

    # Preprocess test set
    X_test = pipeline.transform(X_test)

    X_train, y_train, X_test, y_test, pipeline = process_and_pipeline(df_raw)
    # Check pipeline transformations
    assert X_train.shape[1] == X_test.shape[1], "Mismatch in feature dimensions!"
    assert not pd.isnull(X_train).any().any(), "Training set contains NaN after processing!"
    assert not pd.isnull(X_test).any().any(), "Test set contains NaN after processing!"