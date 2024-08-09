from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import os

app = FastAPI()

# Load the model with error handling
class Wine(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict/")
def predict(wine: Wine):
    model = None
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mlp_model_path = os.path.join(dir_path, "mlp_model.pkl")
        model = joblib.load(mlp_model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        data = np.array([[wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid,
                          wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide,
                          wine.total_sulfur_dioxide, wine.density, wine.pH, 
                          wine.sulphates, wine.alcohol]])
        prediction = model.predict(data)
        return {"quality": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/retrain/")
def retrain():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    winequality_red_path = os.path.join(dir_path, "winequality-red.csv")
    winequality_white_path = os.path.join(dir_path, "winequality-white.csv")

    # Load and preprocess new data
    if not os.path.exists(winequality_red_path) or not os.path.exists(winequality_white_path):
        raise HTTPException(status_code=404, detail="CSV files not found.")

    red_wine = pd.read_csv(winequality_red_path, sep=';')
    white_wine = pd.read_csv(winequality_white_path, sep=';')

    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    combined_wine = pd.concat([red_wine, white_wine], ignore_index=True)
    X = combined_wine.drop(['quality', 'wine_type'], axis=1)
    y = combined_wine['quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Retrain the model
    model = MLPClassifier()
    model.fit(X_scaled, y)

    # Save the updated model
    joblib.dump(model, "mlp_model.pkl")

    return {"message": "Model retrained successfully"}

