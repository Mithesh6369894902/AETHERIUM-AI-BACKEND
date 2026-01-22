from fastapi import APIRouter, Depends
from auth import verify_api_key
from modules.modelcraftx.core import run_benchmark

router = APIRouter(prefix="/modelcraft", tags=["ModelCraft-X"])

@router.post("/benchmark")
def benchmark(payload: dict, dep=Depends(verify_api_key)):
    return run_benchmark(payload)

