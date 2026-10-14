"""Diabetes Risk Prediction API — trains on first start, then loads from cache."""
import time
import warnings
from collections import deque
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CACHE = Path(__file__).parent / "model_cache.joblib"
COLUMN_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURE_COLUMNS = COLUMN_NAMES[:-1]
TARGET_COLUMN = "Outcome"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"


class FeatureEngineer:
    def __init__(self, n_pca_components=4, n_clusters=3, random_state=42):
        self.n_pca_components = n_pca_components
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    def _clean(self, frame):
        frame = frame.copy()
        for col in ZERO_AS_MISSING:
            frame[col] = frame[col].replace(0, np.nan)
        return frame

    def fit_transform(self, frame):
        frame = self._clean(frame)
        X_scaled = self.scaler.fit_transform(self.imputer.fit_transform(frame[FEATURE_COLUMNS]))
        return self._assemble(frame, X_scaled, self.pca.fit_transform(X_scaled), self.kmeans.fit_predict(X_scaled))

    def transform(self, frame):
        frame = self._clean(frame)
        X_scaled = self.scaler.transform(self.imputer.transform(frame[FEATURE_COLUMNS]))
        return self._assemble(frame, X_scaled, self.pca.transform(X_scaled), self.kmeans.predict(X_scaled))

    def _assemble(self, frame, X_scaled, pca_feats, clusters):
        out = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS, index=frame.index)
        for i in range(self.n_pca_components):
            out[f"pca_{i+1}"] = pca_feats[:, i]
        out["risk_cluster"] = clusters.astype(float)
        if TARGET_COLUMN in frame.columns:
            out[TARGET_COLUMN] = frame[TARGET_COLUMN].values
        return out


def train():
    print("Training models (first run only, ~30s)...")
    df = pd.read_csv(DATA_URL, names=COLUMN_NAMES)
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df[TARGET_COLUMN], random_state=RANDOM_STATE)
    fe = FeatureEngineer(random_state=RANDOM_STATE)
    train_feat = fe.fit_transform(train_df)
    feature_cols = [c for c in train_feat.columns if c != TARGET_COLUMN]
    X, y = train_feat[feature_cols], train_feat[TARGET_COLUMN]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    spaces = {
        "logistic_regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                                {"C": [0.01, 0.1, 1.0, 10.0], "class_weight": [None, "balanced"]}),
        "random_forest": (RandomForestClassifier(random_state=RANDOM_STATE),
                          {"n_estimators": [200], "max_depth": [None, 6, 10],
                           "min_samples_leaf": [1, 2, 4], "class_weight": [None, "balanced"]}),
    }
    models = {}
    for name, (est, grid) in spaces.items():
        s = GridSearchCV(est, grid, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X, y)
        models[name] = s.best_estimator_
        print(f"  {name}: CV ROC-AUC = {s.best_score_:.4f}")
    baseline = {n: float(np.mean(m.predict_proba(X)[:, 1])) for n, m in models.items()}
    bundle = {"fe": fe, "models": models, "feature_cols": feature_cols, "baseline": baseline}
    joblib.dump(bundle, CACHE)
    return bundle


bundle = joblib.load(CACHE) if CACHE.exists() else train()
fe, fitted_models = bundle["fe"], bundle["models"]
feature_cols, _baseline = bundle["feature_cols"], bundle["baseline"]
_monitor_log = {n: deque(maxlen=200) for n in fitted_models}

app = FastAPI(title="Diabetes Risk Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PatientRecord(BaseModel):
    Pregnancies: float = Field(..., ge=0)
    Glucose: float = Field(..., ge=0)
    BloodPressure: float = Field(..., ge=0)
    SkinThickness: float = Field(..., ge=0)
    Insulin: float = Field(..., ge=0)
    BMI: float = Field(..., ge=0)
    DiabetesPedigreeFunction: float = Field(..., ge=0)
    Age: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(fitted_models)}


@app.post("/predict")
def predict(record: PatientRecord,
            model_name: Literal["logistic_regression", "random_forest"] = "random_forest"):
    if model_name not in fitted_models:
        raise HTTPException(400, f"Unknown model '{model_name}'")
    start = time.perf_counter()
    raw = pd.DataFrame([record.model_dump()])[FEATURE_COLUMNS]
    feats = fe.transform(raw)
    proba = float(fitted_models[model_name].predict_proba(feats[feature_cols])[0, 1])
    _monitor_log[model_name].append(proba)
    return {"model_used": model_name, "risk_probability": round(proba, 4),
            "risk_label": int(proba >= 0.5),
            "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


@app.get("/monitoring/summary")
def monitoring_summary():
    out = {}
    for name, log in _monitor_log.items():
        if not log:
            out[name] = {"requests_logged": 0}
            continue
        recent = float(np.mean(log))
        drift = round(recent - _baseline[name], 4)
        out[name] = {"requests_logged": len(log), "rolling_mean_probability": round(recent, 4),
                     "training_baseline_mean_probability": round(_baseline[name], 4),
                     "probability_drift": drift, "drift_flag": abs(drift) > 0.15}
    return out
