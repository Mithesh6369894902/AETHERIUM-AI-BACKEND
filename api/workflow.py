from fastapi import APIRouter
from modules.alphaflux.workflow import run_workflow

router = APIRouter()

@router.post("/run")
def run(payload: dict):
    return run_workflow(payload["task"])
