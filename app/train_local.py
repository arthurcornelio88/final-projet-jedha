from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data
from app.model import create_model
from datetime import datetime
import joblib

def train_model(data_file, model_folder):
    df_raw = load_data(data_file)
    print("Data loaded.")

    X_train, y_train, X_test, y_test, pipeline = process_and_pipeline(df_raw, strat=True)
    print("Data processed.")

    # Create and train the model
    model = create_model()
    print("Model created!")

    model.train(X_train, y_train)
    print("Model trained!")

    # Evaluate the model using the custom loss function and RSME
    loss, mae, rmse, w_mse = model.evaluate(X_test, y_test)
    print("Model evaluated!")
    print("Custom loss:", loss)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Weighted MSE: ", w_mse)

    # Save the trained model
    time_creation = datetime.now().strftime('%Y%m%d_%H-%M-%S - ')
    model_name = "trained_model.pkl"
    pipeline_name = "preprocessing_pipeline.pkl"

    model_filepath = model_folder + time_creation + model_name
    pipeline_filepath = model_folder + time_creation + pipeline_name

    model.save(model_filepath)
    joblib.dump(pipeline, pipeline_filepath)  # Save the pipeline
    print(f"Model saved at {model_filepath}")
    print(f"Pipeline saved at {pipeline_filepath}")

if __name__ == "__main__":
    data_file = 'data/total_data.csv'  # Path to your dataset
    model_folder = 'models/' # Folder to save your trained model

    # Train the model and evaluate it
    train_model(data_file, model_folder)
