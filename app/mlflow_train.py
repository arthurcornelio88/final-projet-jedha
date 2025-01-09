from app.pipeline import process_and_pipeline
from app.data_preprocessing import load_data
from app.model import create_model
from sklearn.model_selection import GridSearchCV
import mlflow
import time


def train_model(X_train_processed, y_train, param_grid, cv=2, n_jobs=-1, verbose=3):
    """
    Train the model using GridSearchCV.

    Args:
        X_train_processed (pd.DataFrame): Training processed features.
        y_train (pd.Series): Training target.
        param_grid (dict): The hyperparameter grid to search over.
        cv (int): Number of cross-validation folds.
        n_jobs (int): Number of jobs to run in parallel.
        verbose (int): Verbosity level.

    Returns:
        GridSearchCV: Trained GridSearchCV object.
    """
    tree_model = create_model()
    gs_cv_model = GridSearchCV(tree_model, param_grid, n_jobs=n_jobs, verbose=verbose, cv=cv, scoring="r2")
    gs_cv_model.fit(X_train_processed, y_train)
    return gs_cv_model

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

def run_experiment(experiment_name, data_file, param_grid, artifact_path, registered_model_name):
    """
    Run the entire ML experiment pipeline.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        data_file (str): URL to load the dataset.
        param_grid (dict): The hyperparameter grid for GridSearchCV.
        artifact_path (str): Path to store the model artifact.
        registered_model_name (str): Name to register the model under in MLflow.
    """
    # Start timing
    start_time = time.time()

    # Load, preprocess data and create pipeline
    df_raw = load_data(data_file)
    X_train, X_test, y_train, y_test, pipeline = process_and_pipeline(df_raw)

    # Set experiment's info 
    mlflow.set_experiment(experiment_name)

    # Get our experiment info
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Call mlflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Train model
        model = train_model(X_train, y_train, param_grid)

        # Log metrics and model to MLflow
        log_metrics_and_model(model, X_train, y_train, X_test, y_test, artifact_path, registered_model_name)

    # Print timing
    print(f"...Training Done! --- Total training time: {time.time() - start_time} seconds")

if __name__ == "__main__":
    
    experiment_name = "hyperparameter_tuning"
    data_file = 'data/total_data.csv'  # Path to your dataset
    model_folder = 'models/' # Folder to save your trained model

    param_grid = {
        "max_depth": [3, 5, 10, 15],  # Profondeurs d'arbre à tester
        "min_samples_split": [2, 5, 10, 20],  # Minimum pour diviser un nœud
        "min_samples_leaf": [1, 2, 4, 8],  # Minimum d'échantillons dans une feuille
        "criterion": ["squared_error", "absolute_error"],  # Fonction de perte
        "max_features": [None, "sqrt", "log2"]  # Options de sélection des caractéristiques
    }

    artifact_path = "modeling_airbnb_pricing"
    registered_model_name = "decision_tree"

    # Run the experiment
    run_experiment(experiment_name, data_file, param_grid, artifact_path, registered_model_name)