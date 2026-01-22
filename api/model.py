from fastapi import APIRouter, Depends
from auth import verify_api_key
from logger import log
from modules.modelcraftx.core import run_benchmark

router = APIRouter(prefix="/modelcraft", tags=["ModelCraft-X"])

@router.post("/benchmark")
def benchmark(payload: dict, dep=Depends(verify_api_key)):
    log("ModelCraft-X benchmarking executed")
    return run_benchmark(payload)

