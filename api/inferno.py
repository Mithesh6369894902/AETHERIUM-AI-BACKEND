from fastapi import APIRouter, UploadFile, File, Depends
import pandas as pd
import numpy as np

from auth import verify_api_key

router = APIRouter(prefix="/inferno", tags=["InfernoData"])

# ---------------- DATASET GENERATOR ---------------- #
@router.post("/generate")
def generate_dataset(payload: dict, dep=Depends(verify_api_key)):
    rows = int(payload.get("rows", 100))
    cols = int(payload.get("cols", 4))

    data = np.random.randn(rows, cols)
    df = pd.DataFrame(
        data,
        columns=[f"Feature_{i+1}" for i in range(cols)]
    )

    return df.to_dict(orient="records")

# ---------------- DATASET TRIMMER ---------------- #
@router.post("/trim")
def trim_dataset(
    file: UploadFile = File(...),
    dep=Depends(verify_api_key)
):
    df = pd.read_csv(file.file)
    return df.head(100).to_dict(orient="records")

# ---------------- CLASSIFICATION ---------------- #
@router.post("/classify")
def classify_dataset(
    file: UploadFile = File(...),
    target: str = "",
    dep=Depends(verify_api_key)
):
    df = pd.read_csv(file.file)

    if target not in df.columns:
        return {"error": "Invalid target column"}

    return {
        "task": "classification",
        "rows": df.shape[0],
        "features": list(df.columns),
        "target": target
    }

# ---------------- REGRESSION ---------------- #
@router.post("/regress")
def regress_dataset(
    file: UploadFile = File(...),
    target: str = "",
    dep=Depends(verify_api_key)
):
    df = pd.read_csv(file.file)

    if target not in df.columns:
        return {"error": "Invalid target column"}

    return {
        "task": "regression",
        "rows": df.shape[0],
        "features": list(df.columns),
        "target": target
    }

# ---------------- CLUSTERING ---------------- #
@router.post("/cluster")
def cluster_dataset(
    file: UploadFile = File(...),
    k: int = 3,
    dep=Depends(verify_api_key)
):
    df = pd.read_csv(file.file)

    return {
        "task": "clustering",
        "rows": df.shape[0],
        "clusters": k
    }

# ---------------- ASSOCIATION RULES ---------------- #
@router.post("/associate")
def association_rules(
    file: UploadFile = File(...),
    dep=Depends(verify_api_key)
):
    df = pd.read_csv(file.file)

    return {
        "task": "association",
        "transactions": df.shape[0]
    }
