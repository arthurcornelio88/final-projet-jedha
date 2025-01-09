from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np

class Model:
    def __init__(self):
        self.model = DecisionTreeRegressor( # with slight hyperparametrization
            max_depth=10, 
            min_samples_split=20,
            random_state=42) 

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        # Compute loss
        loss = custom_loss_function(y_test, predictions)
        # Compute MAE
        mae = mae_function(y_test, predictions)
        # Compute RSME
        rmse = rsme_function(y_test, predictions)
        # Compute weighted MSE
        w_mse = weighted_mse(y_test, predictions)

        return loss, mae, rmse, w_mse

    def save(self, file_path):
        save_model(self.model, file_path)

    def load(self, file_path):
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
