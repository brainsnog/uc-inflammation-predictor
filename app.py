"""
app.py
------
UC Histological Inflammation Severity Predictor
FastAPI application — serves prediction API and web interface.

Endpoints:
    GET  /          → HTML prediction interface
    POST /predict   → JSON prediction endpoint
    GET  /health    → health check for deployment platforms
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel, Field, validator
from typing import Optional

# ─────────────────────────────────────────────
# STARTUP — load models once at boot
# ─────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_models():
    """Load serialized models and metadata from disk."""
    try:
        regressor  = joblib.load(os.path.join(MODEL_DIR, 'rf_regressor.pkl'))
        classifier = joblib.load(os.path.join(MODEL_DIR, 'rf_classifier.pkl'))
        meta       = joblib.load(os.path.join(MODEL_DIR, 'model_meta.pkl'))
        return regressor, classifier, meta
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Model files not found in {MODEL_DIR}. "
            f"Run train.py first to generate .pkl files.\n{e}"
        )

rf_regressor, rf_classifier, model_meta = load_models()

FEATURES          = model_meta['features']
BINARY_THRESHOLD  = model_meta['binary_threshold']

# ─────────────────────────────────────────────
# FEATURE DISPLAY METADATA
# Human-readable labels and clinical descriptions
# for each of the 8 RF-selected features
# ─────────────────────────────────────────────

FEATURE_META = {
    'neutro_pereosino_1_std_CALC': {
        'label':       'Neutrophil-per-Eosinophil Ratio Variability (Region 1)',
        'description': 'Standard deviation of neutrophil-to-eosinophil spatial ratio in neighbourhood region 1',
        'unit':        'ratio SD',
        'min': 0.0, 'max': 10.0, 'step': 0.001,
    },
    'neutro_perneutro_1_CALC': {
        'label':       'Neutrophil Self-Neighbour Ratio (Region 1)',
        'description': 'Mean proportion of neutrophil neighbours around each neutrophil in region 1',
        'unit':        'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.001,
    },
    'neutro_perepith_1_std_CALC': {
        'label':       'Neutrophil-per-Epithelial Ratio Variability (Region 1)',
        'description': 'Standard deviation of neutrophil-to-epithelial spatial ratio in neighbourhood region 1',
        'unit':        'ratio SD',
        'min': 0.0, 'max': 10.0, 'step': 0.001,
    },
    'neutro_pereosino_2_CALC': {
        'label':       'Neutrophil-per-Eosinophil Ratio (Region 2)',
        'description': 'Mean proportion of eosinophil neighbours around each neutrophil in region 2',
        'unit':        'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.001,
    },
    'RegionFTotalNorm_CALC': {
        'label':       'Region F Normalised Cell Density (Total)',
        'description': 'Total normalised cell density in anatomical Region F of the colon',
        'unit':        'cells/area (normalised)',
        'min': 0.0, 'max': 5.0, 'step': 0.0001,
    },
    'neutro_pereosino_1_CALC': {
        'label':       'Neutrophil-per-Eosinophil Ratio (Region 1)',
        'description': 'Mean proportion of eosinophil neighbours around each neutrophil in region 1',
        'unit':        'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.001,
    },
    'neutro_perother_2_CALC': {
        'label':       'Neutrophil-per-Other Cell Ratio (Region 2)',
        'description': 'Mean proportion of other cell-type neighbours around each neutrophil in region 2',
        'unit':        'ratio',
        'min': 0.0, 'max': 1.0, 'step': 0.001,
    },
    'NeutroRegionA_CALC': {
        'label':       'Neutrophil Count — Region A',
        'description': 'Absolute neutrophil cell count in anatomical Region A of the colon',
        'unit':        'cell count',
        'min': 0.0, 'max': 500.0, 'step': 0.1,
    },
}

# ─────────────────────────────────────────────
# PYDANTIC INPUT SCHEMA
# ─────────────────────────────────────────────

class PredictionInput(BaseModel):
    neutro_pereosino_1_std_CALC: float = Field(..., ge=0.0, description="Neutrophil-per-Eosinophil ratio variability, region 1")
    neutro_perneutro_1_CALC:     float = Field(..., ge=0.0, description="Neutrophil self-neighbour ratio, region 1")
    neutro_perepith_1_std_CALC:  float = Field(..., ge=0.0, description="Neutrophil-per-Epithelial ratio variability, region 1")
    neutro_pereosino_2_CALC:     float = Field(..., ge=0.0, description="Neutrophil-per-Eosinophil ratio, region 2")
    RegionFTotalNorm_CALC:       float = Field(..., ge=0.0, description="Region F normalised total cell density")
    neutro_pereosino_1_CALC:     float = Field(..., ge=0.0, description="Neutrophil-per-Eosinophil ratio, region 1")
    neutro_perother_2_CALC:      float = Field(..., ge=0.0, description="Neutrophil-per-other-cell ratio, region 2")
    NeutroRegionA_CALC:          float = Field(..., ge=0.0, description="Neutrophil count in Region A")

    class Config:
        json_schema_extra = {
            "example": {
                "neutro_pereosino_1_std_CALC": 0.245,
                "neutro_perneutro_1_CALC":     0.312,
                "neutro_perepith_1_std_CALC":  0.189,
                "neutro_pereosino_2_CALC":     0.078,
                "RegionFTotalNorm_CALC":       0.421,
                "neutro_pereosino_1_CALC":     0.156,
                "neutro_perother_2_CALC":      0.203,
                "NeutroRegionA_CALC":          12.5,
            }
        }

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────

def run_prediction(input_data: PredictionInput) -> dict:
    """Run both regressor and classifier, return structured result."""

    # Build feature array in the exact order the models were trained on
    X = np.array([[getattr(input_data, f) for f in FEATURES]])

    # Continuous severity score (0–5)
    severity_score = float(rf_regressor.predict(X)[0])
    severity_score = round(np.clip(severity_score, 0.0, 5.0), 3)

    # Binary classification
    inflamed_prob  = float(rf_classifier.predict_proba(X)[0][1])
    is_inflamed    = bool(rf_classifier.predict(X)[0])

    # Ordinal label derived from continuous score
    score_rounded = round(severity_score)
    ordinal_labels = {
        0: "No inflammation",
        1: "Minimal inflammation",
        2: "Mild inflammation",
        3: "Moderate inflammation",
        4: "Marked inflammation",
        5: "Severe inflammation",
    }

    return {
        'severity_score':    severity_score,
        'severity_label':    ordinal_labels.get(score_rounded, "Unknown"),
        'is_inflamed':       is_inflamed,
        'inflamed_label':    "Inflamed" if is_inflamed else "Not Inflamed",
        'inflamed_prob':     round(inflamed_prob * 100, 1),
        'binary_threshold':  BINARY_THRESHOLD,
    }

# ─────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="UC Inflammation Severity Predictor",
    description=(
        "Predicts histological inflammation severity in ulcerative colitis "
        "from quantitative tissue features using Random Forest models trained "
        "on 303 assessments from 106 patients under patient-level cross-validation."
    ),
    version="1.0.0",
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), 'templates'))


@app.get("/health")
def health_check():
    """Health check endpoint — used by Render and other platforms."""
    return {"status": "ok", "model": "rf_regressor + rf_classifier", "features": len(FEATURES)}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the prediction interface."""
    return templates.TemplateResponse("index.html", {
        "request":      request,
        "features":     FEATURES,
        "feature_meta": FEATURE_META,
    })


@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Accept feature values and return severity prediction.

    Returns:
        severity_score:  continuous prediction 0–5
        severity_label:  ordinal descriptor
        is_inflamed:     boolean (score >= 3)
        inflamed_label:  'Inflamed' or 'Not Inflamed'
        inflamed_prob:   classifier confidence %
    """
    try:
        result = run_prediction(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─────────────────────────────────────────────
# LOCAL DEV ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)