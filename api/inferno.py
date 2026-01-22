from fastapi import APIRouter
from modules.infernodata.core import preprocess_data

router = APIRouter()

@router.post("/preprocess")
def preprocess(data: list):
    return preprocess_data(data)
