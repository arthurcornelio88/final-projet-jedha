from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import mlflow.sklearn
import numpy as np

class Model(BaseEstimator, RegressorMixin):
    def __init__(self, model=None, max_depth=10, min_samples_split=20, random_state=42, criterion="squared_error", min_samples_leaf=1, max_features=None):
        """
        Initialize the custom Model class.

        Args:
            model (object): An externally provided model (e.g., from RandomizedSearchCV). If None, a DecisionTreeRegressor is created.
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            random_state (int): The random state for reproducibility.
            criterion (str): The function to measure the quality of a split.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str or int): The number of features to consider when looking for the best split.
        """
        self.model = model  # Use external model if provided
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        if self.model is None:  # Initialize DecisionTreeRegressor if no external model is provided
            self.model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
        self.model.fit(X, y)
        return self

    def train(self, X, y):
        """
        Train the model (alias for fit).
        """
        self.fit(X, y)

    def score(self, X, y):
        """
        Calculate the R2 score.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or trained.")
        return self.model.score(X, y)

    def predict(self, X):
        """
        Predict using the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or trained.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        """
        predictions = self.predict(X_test)
        loss = custom_loss_function(y_test, predictions)
        mae = mae_function(y_test, predictions)
        rmse = rsme_function(y_test, predictions)
        w_mse = weighted_mse(y_test, predictions)
        return loss, mae, rmse, w_mse

    def save(self, file_path):
        """
        Save the model to a file.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or trained.")
        save_model(self.model, file_path)

    def load(self, file_path):
        """
        Load the model from a file.
        """
        self.model = load_model(file_path)

def create_model():
    return Model()

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)

def custom_loss_function(y_true, y_pred):
    return sum(np.subtract(y_true, y_pred) ** 2) / len(y_true)

def rsme_function(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse) # return RSME - Root square mean error

def mae_function(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return mae# return MAE - mean absolute error

def weighted_mse(y_true, y_pred):
    weights = 1 / (1 + y_true)  # Inverse of price, adjust as needed
    return np.mean(weights * (y_true - y_pred)**2)

def load_model(model_placement, model_uri=None, model_path=None):
    """
    Load a machine learning model from either MLflow or a local file.

    Args:
        model_placement (str): The source of the model. Options are "mlflow" or "local".
        model_uri (str, optional): The URI of the model in MLflow. Required if `model_placement` is "mlflow".
        model_path (str, optional): The local path to the model file. Required if `model_placement` is "local".

    Returns:
        model: The loaded model instance.

    Raises:
        ValueError: If the arguments are missing or incorrectly specified.
    """

    # Check for MLflow model loading
    if model_placement == "mlflow":
        if model_uri is not None:
            try:
                model = mlflow.sklearn.load_model(model_uri)
                print(f"Model successfully loaded from MLflow: {model_uri}")
                return model
            except Exception as e:
                raise ValueError(f"Failed to load the model from MLflow at {model_uri}. Error: {e}")
        else:
            raise ValueError("Missing `model_uri`. You must specify the MLflow model URI when using 'mlflow' as the placement.")

    # Check for local model loading
    elif model_placement == "local":
        if model_path is not None:
            try:
                model = joblib.load(model_path)
                print(f"Model successfully loaded from local path: {model_path}")
                return model
            except Exception as e:
                raise ValueError(f"Failed to load the model from local path {model_path}. Error: {e}")
        else:
            raise ValueError("Missing `model_path`. You must specify the local model file path when using 'local' as the placement.")

    # Handle invalid `model_placement` values
    else:
        raise ValueError("Invalid `model_placement`. Use 'mlflow' for MLflow models or 'local' for locally saved models.")
