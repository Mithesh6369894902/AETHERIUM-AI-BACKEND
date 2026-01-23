# modelcraft.py
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder

router = APIRouter(prefix="/modelcraft", tags=["ModelCraft-X"])

# ---------- Simple API key verification ----------

API_KEY_VALUE = "my-secret-key"  # set this and match it in Streamlit


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY_VALUE:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


# ---------- Request / response models ----------

class BenchmarkRequest(BaseModel):
    data: List[Dict[str, Any]]
    target: str


class BenchmarkResult(BaseModel):
    best_model: str
    metric: str
    final_score: float
    benchmark: List[Dict[str, Any]]


# ---------- Helper: simple AutoML benchmarking ----------

def prepare_data(df: pd.DataFrame, target: str):
    """Encode categoricals and split X, y."""
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataset")

    y = df[target]
    X = df.drop(columns=[target])

    # Simple label encoding for categoricals
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if classification
    is_classification = y.dtype == object or y.nunique() <= 20
    if is_classification:
        y = LabelEncoder().fit_transform(y.astype(str))

    return X, y, is_classification


@router.post("/benchmark", response_model=BenchmarkResult)
def benchmark_endpoint(payload: BenchmarkRequest, dep=Depends(verify_api_key)):
    # Convert incoming data to DataFrame
    df = pd.DataFrame(payload.data)

    X, y, is_classification = prepare_data(df, payload.target)

    if is_classification:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
        }
        metric = "accuracy"
        scorer = get_scorer(metric)
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100),
        }
        metric = "r2"
        scorer = get_scorer(metric)

    results = []
    best_model_name = None
    best_score = -np.inf

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=3, scoring=scorer)
            mean_score = float(np.mean(scores))
        except Exception:
            # Fallback in case a model fails
            mean_score = float("nan")

        results.append({"model": name, "score": mean_score})

        if not np.isnan(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_model_name = name

    if best_model_name is None:
        best_model_name = "No valid model"
        best_score = float("nan")

    return BenchmarkResult(
        best_model=best_model_name,
        metric=metric,
        final_score=best_score,
        benchmark=results,
    )
