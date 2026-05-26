'''FastAPI backend (STEP 7).'''

from __future__ import annotations

import os
from typing import Any, Dict, List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.preprocessing import prepare_features

app = FastAPI(title="Employee Attrition API")


class PredictRequest(BaseModel):
    # A single employee record as provided by the dataset.
    # Keys should match the original raw feature column names.
    # (Attrition label must NOT be included.)
    data: Dict[str, Any] = Field(..., description="Raw employee features")


MODEL_PATH = os.environ.get("MODEL_PATH", "models/attrition_pipeline.joblib")

# Ensure model path is relative to project root when running under different working dirs
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(_BASE_DIR, MODEL_PATH)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Train the model (python src/models/train.py) to generate it."
        )
    return joblib.load(MODEL_PATH)


_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        # prepare_features expects a DataFrame with the full raw columns.
        row = req.data

        # Validate required columns exist based on preprocessing definition.
        df = prepare_features(
            # prepare_features internally selects the exact needed columns,
            # but it expects they exist in the incoming df.
            # Provide a single-row DF with all raw columns.
            # If a required column is missing, KeyError will be raised.
            #
            # Create temp full df by passing a 1-row DataFrame.
            # (We can't import pandas here unless needed; however sklearn preprocessing
            # pipeline handles DataFrame/array; prepare_features uses pandas.)
            __import__("pandas").DataFrame([row])
        )

        proba = model.predict_proba(df)[:, 1][0]
        pred = int(proba >= 0.5)

        # Also provide label form for readability.
        label = "Yes" if pred == 1 else "No"

        return {
            "prediction": pred,
            "prediction_label": label,
            "probability_yes": float(proba),
            "model": os.path.basename(MODEL_PATH),
        }
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature(s): {str(e)}",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


