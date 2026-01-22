from fastapi import APIRouter, Depends
from auth import verify_api_key
from logger import log
from modules.infernodata.core import preprocess_data

router = APIRouter(prefix="/inferno", tags=["InfernoData"])

@router.post("/preprocess")
def preprocess(data: list, dep=Depends(verify_api_key)):
    log("InfernoData preprocessing called")
    return preprocess_data(data)
