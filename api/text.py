from fastapi import APIRouter, Depends
from auth import verify_api_key
from logger import log
from modules.textvortex.core import sentiment_analysis

router = APIRouter(prefix="/text", tags=["TextVortex"])

@router.post("/sentiment")
def sentiment(payload: dict, dep=Depends(verify_api_key)):
    log("TextVortex sentiment analysis executed")
    return {
        "sentiment": sentiment_analysis(payload["text"])
    }
