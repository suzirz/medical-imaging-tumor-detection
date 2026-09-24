import cv2
import numpy as np
from PIL import Image

def crop_brain_contour(image: np.ndarray, plot: bool = False) -> np.ndarray:
    """
    Crops the brain region from an MRI scan image based on contour detection
    (as implemented by MohamedAliHabib).
    
    Steps:
    1. Convert to Grayscale & Gaussian Blur
    2. Threshold to binary mask
    3. Find the extreme contours of the brain (top, bottom, left, right)
    4. Crop strictly to the brain area, eliminating black background padding
    """
    # 1. Convert to grayscale if RGB
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
        
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Binary threshold
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # 3. Find largest contour
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return image

    c = max(cnts, key=cv2.contourArea)

    # 4. Find extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # 5. Crop
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    if new_image.size == 0:
        return image
    return new_image

def preprocess_mri_240(image: Image.Image) -> np.ndarray:
    """
    Standard preprocessing matching MohamedAliHabib's pipeline:
    Crop Brain Contour -> Resize to (240, 240, 3) -> Normalize to [0, 1].
    """
    img_np = np.array(image.convert("RGB"))
    cropped = crop_brain_contour(img_np)
    resized = cv2.resize(cropped, (240, 240), interpolation=cv2.INTER_CUBIC)
    normalized = resized.astype(np.float32) / 255.0
    return normalized
