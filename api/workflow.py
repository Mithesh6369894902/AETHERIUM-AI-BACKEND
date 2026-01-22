from fastapi import APIRouter, Depends
from auth import verify_api_key
from logger import log
from modules.alphaflux.workflow import run_forecast

router = APIRouter(prefix="/workflow", tags=["AlphaFlux"])

@router.post("/alphaflux/forecast")
def forecast(payload: dict, dep=Depends(verify_api_key)):
    log("AlphaFlux forecasting executed")
    return run_forecast(payload)
