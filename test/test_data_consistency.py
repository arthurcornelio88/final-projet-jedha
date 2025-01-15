from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data
import numpy as np

def test_data_consistency(csv_file):
    #data_file = 'data/total_data.csv'
    df_raw_1 = load_data(csv_file)
    df_raw_2 = load_data(csv_file)

    # Process data
    X_train_1, y_train_1, X_test_1, y_test_1, _ = process_and_pipeline(df_raw_1)
    X_train_2, y_train_2, X_test_2, y_test_2, _ = process_and_pipeline(df_raw_2)

    # Check that transformations are deterministic
    assert np.array_equal(X_train_1, X_train_2), "Processed training data is not consistent!"
    assert np.array_equal(X_test_1, X_test_2), "Processed test data is not consistent!"
