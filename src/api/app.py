from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Risk PD API")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "dynamic_model.joblib"
META_PATH = ROOT / "models" / "meta.json"

model = None
meta = None


@app.on_event("startup")
def load_artifacts():
    global model, meta
    try:
        model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            import json
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")


class PredictRequest(BaseModel):
    current_state: int = Field(..., ge=0, le=2)

    # Признаки (должны совпадать с FEATURE_COLS)
    util_mean: float
    util_std: float
    util_max: float
    util_min: float
    util_range: float
    util_trend: float
    util_mean_last3: float
    util_mean_prev: float
    util_accel: float
    util_growth_ratio: float
    n_state_changes: int
    state_change_rate: float
    frac_current: float
    frac_delinquent: float
    frac_default: float
    max_consec_delinquent: int
    months_since_last_del: int
    state_entropy: float
    util_cv: float
    high_util_ratio: float
    ever_delinquent: int
    util_rising_3m: int
    util_vs_mean_dev: float
    utilization: float
    macro_stress: float
    util_x_macro: float




@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        x = to_feature_vector(req)
        state = req.current_state

        # Предполагаем, что модель умеет считать PD
        # Замени на свой метод (score_one / predict_proba и т.д.)
        pd_1m = float(model.predict_proba_row(state, x)[2])  # пример

        return {
            "pd_1m": round(pd_1m, 5),
            "current_state": state,
            "model_version": meta.get("version", "unknown") if meta else "unknown",
        }
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ready",
        "model_version": meta.get("version", "unknown") if meta else "unknown",
    }