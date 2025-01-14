from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data, load_and_sample_data
from app.model import Model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import mlflow
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

# Log metrics and model to MLflow
def log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name):
    """
    Log training and test metrics, and the model to MLflow.
    Args:
        model (GridSearchCV): The trained model.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """

    mlflow.log_metric("Train Score", model.score(X_train, y_train))
    mlflow.log_metric("Test Score", model.score(X_test, y_test))
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name
    )

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

    # Load sampled data. Comment when loading all dataset
    # print("Loading sample data...")
    # df_raw = load_and_sample_data(data_file, sample_fraction=0.3)
    # Verify the experiment was retrieved successfully

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found. Please create it in MLflow.")

    # Uncomment when loading all dataset
    print("Loading data...")
    df_raw = load_data(data_file)

    # preprocess data and create pipeline
    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, pipeline = process_and_pipeline(df_raw, strat=True)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Train model with RandomizedSearchCV
    print("Training model...")
    best_model = train_model(X_train, y_train, param_distributions, n_iter=2)

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):

        # Train model with RandomizedSearch
        best_model = train_model(X_train, y_train, param_distributions, n_iter=2).best_estimator_

        # Log metrics and model to MLflow
        # log_metrics_and_model(best_model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)

    # Print timing
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

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
