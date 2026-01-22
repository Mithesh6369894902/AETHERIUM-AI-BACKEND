from fastapi import APIRouter
from modules.textvortex.core import sentiment_analysis

router = APIRouter()

@router.post("/sentiment")
def sentiment(payload: dict):
    return {"sentiment": sentiment_analysis(payload["text"])}
