import cv2
import numpy as np

def analyze_image(image_bytes: bytes):
    image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    height, width, channels = img.shape
    return {
        "width": width,
        "height": height,
        "channels": channels
    }
