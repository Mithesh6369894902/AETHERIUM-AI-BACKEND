import cv2
import numpy as np
from fastapi.responses import Response

def edge_detection(image_bytes: bytes, t1: int = 80, t2: int = 180):
    """
    Performs Canny edge detection and returns PNG image bytes.
    """
    # Convert bytes → OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"", status_code=400)

    edges = cv2.Canny(img, t1, t2)

    # Encode back to PNG
    success, buffer = cv2.imencode(".png", edges)
    if not success:
        return Response(content=b"", status_code=500)

    return Response(
        content=buffer.tobytes(),
        media_type="image/png"
    )
