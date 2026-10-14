import cv2
import numpy as np
from typing import Tuple, Optional

class RooftopImagePreprocessor:
    """
    OpenCV-based image preprocessing pipeline for aerial and satellite rooftop imagery.
    """
    def __init__(self, target_size: Tuple[int, int] = (512, 512), blur_kernel: int = 5):
        self.target_size = target_size
        self.blur_kernel = blur_kernel

    def load_and_standardize(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image from {image_path}")
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        return resized

    def preprocess_for_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Adaptive thresholding for handling varying solar reflection
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to bridge broken roof outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.Canny(blurred, 50, 150)
        
        return closed, edges
