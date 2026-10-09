
from pathlib import Path
import hashlib
import json
from typing import Annotated

import numpy as np
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_random_forest.joblib"
metadata = json.loads((BASE_DIR / "metadata.json").read_text(encoding="utf-8"))
model = load(MODEL_PATH)
model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
if model_sha256 != metadata["model_sha256"]:
    raise RuntimeError("Model and metadata do not match")

app = FastAPI(title="Iris Prediction API", version="1.0.0",
              description="Educational Random Forest deployment on Render")
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]

class IrisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sepal_length: PositiveFinite
    sepal_width: PositiveFinite
    petal_length: PositiveFinite
    petal_width: PositiveFinite

@app.get("/")
def root():
    return {"message": "Iris Prediction API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "model_version": metadata["model_version"],
            "model_sha256": model_sha256}

@app.post("/predict")
def predict(data: IrisInput):
    payload = data.model_dump()
    features = np.asarray([[payload[key] for key in metadata["feature_order"]]])
    predicted = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    names = metadata["target_names"]
    return {
        "ID": "683380414-8",
        "model_version": metadata["model_version"],
        "input": payload,
        "predicted_class_index": predicted,
        "predicted_class_name": names[predicted],
        "probabilities": {names[int(k)]: float(p)
                          for k, p in zip(model.classes_, probabilities)}
    }
