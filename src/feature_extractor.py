import cv2
import numpy as np
from typing import Dict, Any, List

class RooftopFeatureExtractor:
    """
    Extracts geometric, surface, and orientation features from segmented rooftop contours.
    """
    def __init__(self, pixels_per_meter_ratio: float = 12.5):
        self.ppm = pixels_per_meter_ratio  # Calibrated for high-res orthophotos

    def extract_features(self, binary_mask: np.ndarray) -> Dict[str, Any]:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "total_roof_area_m2": 0.0,
                "usable_solar_area_m2": 0.0,
                "aspect_ratio": 1.0,
                "orientation_angle_deg": 0.0,
                "contour_count": 0,
                "perimeter_m": 0.0
            }
        
        # Find primary rooftop contour
        primary_contour = max(contours, key=cv2.contourArea)
        pixel_area = cv2.contourArea(primary_contour)
        perimeter_px = cv2.arcLength(primary_contour, True)
        
        # Convert pixels to metric (m^2)
        real_area_m2 = pixel_area / (self.ppm ** 2)
        perimeter_m = perimeter_px / self.ppm
        
        # Bounding box & aspect ratio
        rect = cv2.minAreaRect(primary_contour)
        (_, (w, h), angle) = rect
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        # Account for typical setbacks, skylights & HVAC obstacles (~75% usable)
        usable_solar_area_m2 = round(real_area_m2 * 0.75, 2)
        
        return {
            "total_roof_area_m2": round(real_area_m2, 2),
            "usable_solar_area_m2": usable_solar_area_m2,
            "aspect_ratio": round(aspect_ratio, 2),
            "orientation_angle_deg": round(angle, 2),
            "contour_count": len(contours),
            "perimeter_m": round(perimeter_m, 2)
        }
