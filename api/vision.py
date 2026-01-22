from fastapi import APIRouter, UploadFile, File
from modules.visionblaze.core import analyze_image

router = APIRouter()

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    return analyze_image(content)
