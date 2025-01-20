from mlflow.tracking import MlflowClient
import mlflow
import joblib

def load_pipeline_from_mlflow(run_id, artifact_path):
    """
    Download the preprocessing pipeline from MLflow.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        artifact_uri = client.download_artifacts(run_id, artifact_path)
        pipeline = joblib.load(artifact_uri)
        return pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to load pipeline from MLflow: {e}")

def get_latest_run_id(experiment_name):
    """
    Get the latest run ID for a given experiment name.
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found.")

        # Get the latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        return runs[0].info.run_id if runs else None
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve the latest run ID: {e}")
