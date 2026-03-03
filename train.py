"""
train.py
--------
UC Histological Inflammation Severity Predictor
Training script — run once locally to generate serialized models.

Usage:
    python train.py --data /path/to/CSI_7_MAL_2526_Data.xlsx

Outputs:
    models/rf_regressor.pkl    — RandomForestRegressor (continuous severity 0-5)
    models/rf_classifier.pkl   — RandomForestClassifier (binary: inflamed >= 3)
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# The 8 RF-importance-selected features (from cross-validated selection in research)
RF_FEATURES = [
    'neutro_pereosino_1_std_CALC',
    'neutro_perneutro_1_CALC',
    'neutro_perepith_1_std_CALC',
    'neutro_pereosino_2_CALC',
    'RegionFTotalNorm_CALC',
    'neutro_pereosino_1_CALC',
    'neutro_perother_2_CALC',
    'NeutroRegionA_CALC',
]

TARGET_COL   = 'Severity Score'
GROUP_COL    = 'PatID'
BINARY_THRESHOLD = 3        # score >= 3 → inflamed
MAX_MISSING_FEATURES = 100  # rows with more missing values than this are dropped

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Model hyperparameters — match those used in the research exactly
RF_REG_PARAMS = dict(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
RF_CLF_PARAMS = dict(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load Excel file and apply the same cleaning steps used in research."""
    print(f"\n[1/4] Loading data from: {path}")
    df = pd.read_excel(path)
    print(f"      Raw shape: {df.shape}")

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL]).copy()

    # Drop rows that are essentially empty (missed visits)
    feature_cols = df.columns.difference([GROUP_COL, TARGET_COL])
    df = df[df[feature_cols].isnull().sum(axis=1) < MAX_MISSING_FEATURES]

    # Drop helper columns if they exist
    df = df.drop(columns=[c for c in df.columns if c == 'num_missing'], errors='ignore')

    print(f"      Clean shape: {df.shape}")
    print(f"      Patients: {df[GROUP_COL].nunique()}")
    print(f"      Severity score distribution:\n{df[TARGET_COL].value_counts().sort_index().to_string()}")

    return df


# ─────────────────────────────────────────────
# CROSS-VALIDATION EVALUATION
# ─────────────────────────────────────────────

def cross_validate(df: pd.DataFrame) -> dict:
    """
    Run 5-fold patient-level GroupKFold CV to verify model performance
    matches the research results before committing to serialization.
    """
    print("\n[2/4] Running 5-fold patient-level cross-validation...")

    X      = df[RF_FEATURES]
    y_reg  = df[TARGET_COL]
    y_bin  = (y_reg >= BINARY_THRESHOLD).astype(int)
    groups = df[GROUP_COL]

    gkf = GroupKFold(n_splits=5)

    reg_mae, reg_r2 = [], []
    clf_acc, clf_auc = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y_reg, groups), start=1):
        X_train, X_test     = X.iloc[train_idx], X.iloc[test_idx]
        y_reg_tr, y_reg_te  = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        y_bin_tr, y_bin_te  = y_bin.iloc[train_idx], y_bin.iloc[test_idx]

        # Regression
        rf_r = RandomForestRegressor(**RF_REG_PARAMS)
        rf_r.fit(X_train, y_reg_tr)
        y_reg_pred = rf_r.predict(X_test)
        reg_mae.append(mean_absolute_error(y_reg_te, y_reg_pred))
        reg_r2.append(r2_score(y_reg_te, y_reg_pred))

        # Classification
        rf_c = RandomForestClassifier(**RF_CLF_PARAMS)
        rf_c.fit(X_train, y_bin_tr)
        y_bin_pred  = rf_c.predict(X_test)
        y_bin_proba = rf_c.predict_proba(X_test)[:, 1]
        clf_acc.append(accuracy_score(y_bin_te, y_bin_pred))
        clf_auc.append(roc_auc_score(y_bin_te, y_bin_proba))

        print(f"      Fold {fold}: MAE={reg_mae[-1]:.3f}  R²={reg_r2[-1]:.3f}  "
              f"Acc={clf_acc[-1]:.3f}  AUC={clf_auc[-1]:.3f}")

    results = {
        'mae_mean': np.mean(reg_mae), 'mae_std': np.std(reg_mae),
        'r2_mean':  np.mean(reg_r2),  'r2_std':  np.std(reg_r2),
        'acc_mean': np.mean(clf_acc), 'acc_std': np.std(clf_acc),
        'auc_mean': np.mean(clf_auc), 'auc_std': np.std(clf_auc),
    }

    print(f"\n      ── CV Summary ──────────────────────────────────")
    print(f"      Regression  MAE : {results['mae_mean']:.3f} ± {results['mae_std']:.3f}  "
          f"(paper: 0.684 ± 0.045)")
    print(f"      Regression  R²  : {results['r2_mean']:.3f} ± {results['r2_std']:.3f}  "
          f"(paper: 0.746 ± 0.037)")
    print(f"      Classification Acc : {results['acc_mean']:.3f} ± {results['acc_std']:.3f}  "
          f"(paper: 0.924 ± 0.029)")
    print(f"      Classification AUC : {results['auc_mean']:.3f} ± {results['auc_std']:.3f}  "
          f"(paper: 0.978 ± 0.008)")
    print(f"      ─────────────────────────────────────────────────")

    return results


# ─────────────────────────────────────────────
# FINAL MODEL TRAINING ON FULL DATASET
# ─────────────────────────────────────────────

def train_final_models(df: pd.DataFrame):
    """Train on the complete cleaned dataset for maximum generalization."""
    print("\n[3/4] Training final models on full dataset...")

    X     = df[RF_FEATURES]
    y_reg = df[TARGET_COL]
    y_bin = (y_reg >= BINARY_THRESHOLD).astype(int)

    rf_regressor  = RandomForestRegressor(**RF_REG_PARAMS)
    rf_classifier = RandomForestClassifier(**RF_CLF_PARAMS)

    rf_regressor.fit(X, y_reg)
    rf_classifier.fit(X, y_bin)

    print(f"      Regressor  trained on {len(X)} samples, {len(RF_FEATURES)} features")
    print(f"      Classifier trained on {len(X)} samples, {len(RF_FEATURES)} features")

    return rf_regressor, rf_classifier


# ─────────────────────────────────────────────
# SERIALIZATION
# ─────────────────────────────────────────────

def save_models(rf_regressor, rf_classifier):
    """Serialize models and feature list to the models/ directory."""
    print(f"\n[4/4] Saving models to {MODEL_DIR}/")

    os.makedirs(MODEL_DIR, exist_ok=True)

    reg_path = os.path.join(MODEL_DIR, 'rf_regressor.pkl')
    clf_path = os.path.join(MODEL_DIR, 'rf_classifier.pkl')
    meta_path = os.path.join(MODEL_DIR, 'model_meta.pkl')

    joblib.dump(rf_regressor,  reg_path)
    joblib.dump(rf_classifier, clf_path)

    # Save metadata alongside models so the app can load feature names
    # and display context without hardcoding them in two places
    meta = {
        'features':          RF_FEATURES,
        'target_col':        TARGET_COL,
        'binary_threshold':  BINARY_THRESHOLD,
        'n_features':        len(RF_FEATURES),
    }
    joblib.dump(meta, meta_path)

    print(f"      ✓ {reg_path}")
    print(f"      ✓ {clf_path}")
    print(f"      ✓ {meta_path}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train UC inflammation severity models and serialize to .pkl'
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to CSI_7_MAL_2526_Data.xlsx'
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df                         = load_and_clean(args.data)
    cv_results                 = cross_validate(df)
    rf_regressor, rf_classifier = train_final_models(df)
    save_models(rf_regressor, rf_classifier)

    print("\n✓ Training complete. Ready for deployment.")
    print("  Next step: python app.py  (or docker build)\n")


if __name__ == '__main__':
    main()