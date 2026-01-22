from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np

from auth import verify_api_key

router = APIRouter(prefix="/vision", tags=["VisionBlaze"])

# ---------------- UTIL ---------------- #
def read_image(file: UploadFile):
    contents = file.file.read()
    np_img = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

def send_image(img):
    _, buf = cv2.imencode(".png", img)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/png"
    )

# ---------------- EDGE DETECTION ---------------- #
@router.post("/edges")
def edges(
    file: UploadFile = File(...),
    t1: int = 80,
    t2: int = 180,
    dep=Depends(verify_api_key)
):
    img = read_image(file)
    edges = cv2.Canny(img, t1, t2)
    return send_image(edges)

# ---------------- CONTRAST ENHANCEMENT ---------------- #
@router.post("/contrast")
def contrast(
    file: UploadFile = File(...),
    dep=Depends(verify_api_key)
):
    img = read_image(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    return send_image(enhanced)

# ---------------- SALIENCY MAPPING ---------------- #
@router.post("/saliency")
def saliency(
    file: UploadFile = File(...),
    dep=Depends(verify_api_key)
):
    img = read_image(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return send_image(mag)

# ---------------- SEGMENTATION ---------------- #
@router.post("/segment")
def segment(
    file: UploadFile = File(...),
    dep=Depends(verify_api_key)
):
    img = read_image(file)
    mask = np.zeros(img.shape[:2], np.uint8)

    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    h, w = img.shape[:2]
    rect = (10, 10, w - 20, h - 20)

    cv2.grabCut(img, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    segmented = img * mask2[:, :, np.newaxis]

    return send_image(segmented)
