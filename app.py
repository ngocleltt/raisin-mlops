
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Raisin Classification API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

# Загрузка обученной модели
model = joblib.load(
    BASE_DIR / "models" / "raisin_model.joblib"
)


# Описание входных признаков
class RaisinFeatures(BaseModel):
    Area: float
    MajorAxisLength: float
    MinorAxisLength: float
    Eccentricity: float
    ConvexArea: float
    Extent: float
    Perimeter: float


# Проверка состояния сервиса
@app.get("/")
def health_check():
    return {"status": "ok"}


# Получение предсказания модели
@app.post("/predict")
def predict(features: RaisinFeatures):
    data = pd.DataFrame([features.model_dump()])
    data = data[model.feature_names_in_]

    prediction = model.predict(data)[0]
    probabilities = model.predict_proba(data)[0]

    return {
        "prediction": str(prediction),
        "confidence": float(max(probabilities))
    }
