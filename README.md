# UC Histological Inflammation Severity Predictor

A deployed machine learning application that predicts histological inflammation severity in ulcerative colitis from quantitative tissue features. Built as a full end-to-end ML deployment project — from raw research code to a containerised, publicly accessible web service.

**Live demo:** `[your-render-url-here]`

---

## Project Overview

Histological assessment of ulcerative colitis involves a pathologist assigning an inflammation severity score (0–5) to colon tissue samples. This process is inherently subjective, introducing inter- and intra-observer variability. This project applies feature-based machine learning to predict those scores from 141 quantitative cellular and spatial tissue descriptors, demonstrating that a compact 8-feature model can achieve clinically meaningful accuracy.

The model was trained on a longitudinal dataset of **303 histological assessments from 106 patients** across up to 3 timepoints. All evaluation uses **patient-level GroupKFold cross-validation** to prevent data leakage across longitudinal visits.

---

## Model Performance

Best configuration: **Random Forest + RF-importance-selected features**

| Task | Metric | Score |
|---|---|---|
| Regression | MAE | 0.684 +/- 0.045 |
| Regression | R2 | 0.746 +/- 0.037 |
| Binary Classification | Accuracy | 0.924 +/- 0.029 |
| Binary Classification | ROC-AUC | 0.978 +/- 0.008 |
| Binary Classification | Sensitivity | 0.927 +/- 0.045 |
| Binary Classification | Specificity | 0.926 +/- 0.064 |

Three model types were compared: Random Forest, Support Vector Machine, and Gradient Boosting. Three feature selection strategies were evaluated: manual biology-informed selection, LASSO-based embedded selection, and Random Forest importance-based selection. RF with RF-selected features consistently achieved the strongest performance across all task formulations.

---

## Features Used (8 RF-selected from 141)

| Feature | Description |
|---|---|
| neutro_pereosino_1_std_CALC | Neutrophil-to-eosinophil spatial ratio variability, region 1 |
| neutro_perneutro_1_CALC | Neutrophil self-neighbour ratio, region 1 |
| neutro_perepith_1_std_CALC | Neutrophil-to-epithelial ratio variability, region 1 |
| neutro_pereosino_2_CALC | Neutrophil-to-eosinophil ratio, region 2 |
| RegionFTotalNorm_CALC | Normalised total cell density, Region F |
| neutro_pereosino_1_CALC | Neutrophil-to-eosinophil ratio, region 1 |
| neutro_perother_2_CALC | Neutrophil-to-other-cell ratio, region 2 |
| NeutroRegionA_CALC | Absolute neutrophil count, Region A |

---

## Architecture

```
Request
   |
   v
FastAPI (app.py)
   |
   +-- GET  /          -> Jinja2 HTML interface
   +-- POST /predict   -> Prediction endpoint (returns JSON)
   +-- GET  /health    -> Health check (used by Render)
          |
          v
   RandomForestRegressor   -> continuous severity score (0-5)
   RandomForestClassifier  -> binary label + confidence
```

---

## Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI |
| ML models | scikit-learn RandomForest |
| Model serialization | joblib |
| Templating | Jinja2 |
| Server | Uvicorn |
| Containerization | Docker |
| Deployment | Render |

---

## Local Setup

### Run with Python directly

```bash
git clone https://github.com/your-username/uc-inflammation-predictor.git
cd uc-inflammation-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000

### Run with Docker

```bash
docker build -t uc-predictor .
docker run -p 8000:8000 uc-predictor
```

---

## API Reference

### POST /predict

**Request:**
```json
{
  "neutro_pereosino_1_std_CALC": 0.245,
  "neutro_perneutro_1_CALC": 0.312,
  "neutro_perepith_1_std_CALC": 0.189,
  "neutro_pereosino_2_CALC": 0.078,
  "RegionFTotalNorm_CALC": 0.421,
  "neutro_pereosino_1_CALC": 0.156,
  "neutro_perother_2_CALC": 0.203,
  "NeutroRegionA_CALC": 12.5
}
```

**Response:**
```json
{
  "severity_score": 2.841,
  "severity_label": "Mild inflammation",
  "is_inflamed": false,
  "inflamed_label": "Not Inflamed",
  "inflamed_prob": 31.4,
  "binary_threshold": 3
}
```

Interactive API docs at /docs (FastAPI Swagger UI).

---

## Retraining

The .pkl files in models/ are committed and loaded at runtime. To retrain with the original dataset:

```bash
python train.py --data /path/to/data.xlsx
```

Raw data is excluded from version control via .gitignore.