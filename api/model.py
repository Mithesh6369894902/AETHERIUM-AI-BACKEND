from fastapi import APIRouter
from modules.modelcraftx.core import train_model

router = APIRouter()

@router.post("/train")
def train():
    return train_model()
