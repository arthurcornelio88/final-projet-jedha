from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
from app.pipeline import process_and_pipeline
from app.static_values import TEST_INPUT
import pandas as pd

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle depuis un chemin local ou un URI MLflow
MODEL_PATH = "models/20250109_18-29-55 - trained_model.pkl"  # Ajustez le chemin
MODEL_PLACEMENT = "local"  # "mlflow" ou "local"
model = load_model(model_placement=MODEL_PLACEMENT, model_path=MODEL_PATH)

# Classe de validation des données d'entrée
class PredictionInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Endpoint pour obtenir des prédictions
@app.post("/predictions")
def get_predictions(input: PredictionInput):
    """
    Endpoint pour obtenir des prédictions en fonction des données d'entrée.
    """
    # Convertir les données d'entrée au format DataFrame pour la compatibilité
    input_data = pd.DataFrame([{
        "feature1": input.feature1,
        "feature2": input.feature2,
        "feature3": input.feature3
    }])

    # Appliquer le pipeline pour prétraiter les données
    _, _, _, _, pipeline = process_and_pipeline(input_data, mlflow=None, strat=False)
    prepared_data = pipeline.transform(input_data)

    # Effectuer les prédictions
    prediction = model.predict(prepared_data)
    return {"prediction": prediction.tolist()}

# Endpoint pour tester avec des valeurs statiques
@app.get("/test-prediction")
def test_prediction():
    """
    Endpoint pour tester le modèle avec des valeurs statiques.
    """
    # Convertir les valeurs TEST_INPUT en DataFrame
    test_data = pd.DataFrame([TEST_INPUT])

    # Appliquer le pipeline pour prétraiter les données
    _, _, _, _, pipeline = process_and_pipeline(test_data, mlflow=None, strat=False)
    prepared_data = pipeline.transform(test_data)

    # Effectuer les prédictions
    prediction = model.predict(prepared_data)
    return {"test_input": TEST_INPUT, "prediction": prediction.tolist()}
