from fastapi import APIRouter, Depends, UploadFile, File, Form
from auth import verify_api_key
from logger import log
from modules.visionblaze.core import edge_detection

router = APIRouter(prefix="/vision", tags=["VisionBlaze"])

@router.post("/edges")
async def edges(
    file: UploadFile = File(...),
    t1: int = Form(80),
    t2: int = Form(180),
    dep=Depends(verify_api_key)
):
    log("VisionBlaze edge detection executed")
    image_bytes = await file.read()
    return edge_detection(image_bytes, t1, t2)
