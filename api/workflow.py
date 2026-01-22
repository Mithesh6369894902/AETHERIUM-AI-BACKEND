from fastapi import APIRouter, Depends
from auth import verify_api_key
from modules.alphaflux.workflow import run_forecast

router = APIRouter(prefix="/workflow/alphaflux", tags=["AlphaFlux"])

@router.post("/forecast")
def forecast(payload: dict, dep=Depends(verify_api_key)):
    return run_forecast(payload)
