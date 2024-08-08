# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the model
model = joblib.load("/app/mlp_model.pkl")

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
    data = np.array([[wine.fixed_acidity, wine.volatile_acidity, wine.citric_acid,
                      wine.residual_sugar, wine.chlorides, wine.free_sulfur_dioxide,
                      wine.total_sulfur_dioxide, wine.density, wine.pH, 
                      wine.sulphates, wine.alcohol]])
    prediction = model.predict(data)
    return {"quality": int(prediction[0])}

@app.post("/retrain/")
def retrain():
    # Load and preprocess new data
    red_wine = pd.read_csv('/app/winequality-red.csv', sep=';')
    white_wine = pd.read_csv('/app/winequality-white.csv', sep=';')

    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    combined_wine = pd.concat([red_wine, white_wine], ignore_index=True)
    X = combined_wine.drop(['quality', 'wine_type'], axis=1)
    y = combined_wine['quality']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Retrain the model
    model.fit(X_scaled, y)

    # Save the updated model
    joblib.dump(model, "/app/mlp_model.pkl")

    return {"message": "Model retrained successfully"}
