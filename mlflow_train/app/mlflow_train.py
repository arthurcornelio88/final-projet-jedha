from sklearn.metrics import mean_absolute_error, mean_squared_error
from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data, load_and_sample_data
from app.model import Model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import joblib
import time
import pandas as pd
import numpy as np

def train_model(X_train_processed, y_train, param_distributions, n_iter=10, cv=2, n_jobs=-1, verbose=1, random_state=42):
    """
    Train the model using RandomizedSearchCV for faster hyperparameter optimization.
    Args:
        X_train_processed (np.ndarray): Training processed features.
        y_train (pd.Series): Training target.
        param_distributions (dict): The hyperparameter distribution to search over.
        n_iter (int): Number of random combinations to try.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of jobs to run in parallel.
        verbose (int): Verbosity level.
        random_state (int): Random seed for reproducibility.
    Returns:
        RandomizedSearchCV: Trained RandomizedSearchCV object.
    """
    tree_model = Model()
    n_jobs=1
    rs_cv_model = RandomizedSearchCV(
        estimator=tree_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring="r2",
        random_state=random_state
    )
    rs_cv_model_fitted = rs_cv_model.fit(X_train_processed, y_train.to_numpy())  # Convert y_train to numpy
    return rs_cv_model_fitted

def log_model_and_pipeline_to_mlflow(
    model, pipeline, X_train, y_train, X_test, y_test, artifact_path, registered_model_name
):
    """
    Log the trained model, preprocessing pipeline, and evaluation metrics to MLflow.
    """
    mlflow.sklearn.autolog()  # Enable autologging for additional metrics and parameters

    with mlflow.start_run() as run:
        # Log pipeline as an artifact
        pipeline_path = "pipelines/preprocessing_pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)
        mlflow.log_artifact(pipeline_path, artifact_path="pipeline")

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model.best_estimator_,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )

        # Log evaluation metrics
        y_pred_train = model.best_estimator_.predict(X_train)
        y_pred_test = model.best_estimator_.predict(X_test)

        mlflow.log_metric("Train R2", model.best_estimator_.score(X_train, y_train))
        mlflow.log_metric("Test R2", model.best_estimator_.score(X_test, y_test))
        mlflow.log_metric("Train MAE", mean_absolute_error(y_train, y_pred_train))
        mlflow.log_metric("Test MAE", mean_absolute_error(y_test, y_pred_test))
        mlflow.log_metric("Train RMSE", np.sqrt(mean_squared_error(y_train, y_pred_train)))
        mlflow.log_metric("Test RMSE", np.sqrt(mean_squared_error(y_test, y_pred_test)))

        print(f"Model and pipeline logged to MLflow. Run ID: {run.info.run_id}")
        print(f"Registered Model URI: models:/{registered_model_name}")

def run_experiment(experiment_name, data_file, param_distributions, artifact_path, registered_model_name):
    """
    Run the entire ML experiment pipeline (load data, preprocess, train and log to MLFlow).
    Args:
        experiment_name (str): Name of the MLflow experiment.
        data_file (str): URL to load the dataset.
        param_grid (dict): The hyperparameter grid for GridSearchCV.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    # Start timing
    start_time = time.time()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://mlflow:5000")  # Use the MLflow server in Docker Compose

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Verify the experiment was retrieved successfully
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Please create it in MLflow.")

    # Load sampled data. Comment when loading all dataset
    # print("Loading sample data...")
    # df_raw = load_and_sample_data(data_file, sample_fraction=0.3)

    # Uncomment when loading all dataset
    print("Loading data...")
    df_raw = load_data(data_file)

    # preprocess data and create pipeline
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, pipeline = process_and_pipeline(df_raw, strat=True)

    # Enable MLflow autologging
    mlflow.sklearn.autolog()

    # Train model with RandomizedSearchCV
    print("Training model...")
    best_model = train_model(X_train, y_train, param_distributions, n_iter=5)

    # Log model, pipeline, and metrics to MLflow
    print("Logging to MLflow...")
    log_model_and_pipeline_to_mlflow(
        model=best_model,
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )

    # Print timing
    print(f"Training and logging completed! Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Define experiment and data details
    experiment_name = "hyperparameter_tuning"
    data_file = '/home/data/total_data.csv'  # Update path to match container structure

    # parameters for RandomizedSearch
    param_distributions = {
    "criterion": ["squared_error"],
    "max_depth": [5, 10, None],  # Include None for no limit
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": [1, 4],
    "min_samples_split": [2, 10],
    "random_state": [42],  # Keep fixed for reproducibility
    }

    artifact_path = "modeling_airbnb_pricing"
    registered_model_name = "decision_tree"

    # Run the experiment
    run_experiment(experiment_name, data_file, param_distributions, artifact_path, registered_model_name)
