import pandas as pd
from app.data_preprocessing import load_data

def test_csv_loading():
    data_file = 'data/total_data.csv'

    # Load the dataset
    df_raw = load_data(data_file)
    
    # Check that the dataframe has at least one column and one row
    assert df_raw.shape[0] > 0, "Dataset should have at least one row."
    assert df_raw.shape[1] > 0, "Dataset should have at least one column."

    # Optionally, log if the dataset is entirely NaN
    assert not df_raw.isnull().all(axis=0).all(), "All columns in the dataset are completely NaN!"
    assert not df_raw.isnull().all(axis=1).all(), "All rows in the dataset are completely NaN!"
