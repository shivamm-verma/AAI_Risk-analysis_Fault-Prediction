from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model_registry import ModelRegistry


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = (BASE_DIR.parent / "Model").resolve()

app = FastAPI(
    title="AAI Risk Analysis API",
    version="1.0.0",
)

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:5000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry(MODEL_DIR)


class PredictionResponse(BaseModel):
    model: str
    rows: int
    prediction: List[Any]
    summary: str
    risk_level: str | None = None


@app.on_event("startup")
def load_models() -> None:
    registry.load_all()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> Dict[str, List[str]]:
    return {"models": registry.list_names()}


@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(model_name: str, file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    model = registry.get(model_name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {registry.list_names()}",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV content.")

    try:
        predictions = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    predictions_list = getattr(predictions, "tolist", lambda: list(predictions))()

    risk_level = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df)
            avg_max = float(proba.max(axis=1).mean())
            if avg_max >= 0.8:
                risk_level = "High"
            elif avg_max >= 0.5:
                risk_level = "Medium"
            else:
                risk_level = "Low"
        except Exception:
            pass

    summary = f"Generated {len(predictions_list)} predictions using '{model_name}'."

    return PredictionResponse(
        model=model_name,
        rows=len(df),
        prediction=predictions_list,
        summary=summary,
        risk_level=risk_level,
    )
