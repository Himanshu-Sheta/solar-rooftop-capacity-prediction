import os
import argparse
import json
import numpy as np
from src.preprocessing import RooftopImagePreprocessor
from src.feature_extractor import RooftopFeatureExtractor
from src.model import SolarCapacityEstimator

def run_pipeline(image_path: str = None, irradiance: float = 1150.0, tilt: float = 30.0) -> dict:
    preprocessor = RooftopImagePreprocessor()
    extractor = RooftopFeatureExtractor()
    estimator = SolarCapacityEstimator()
    
    if image_path and os.path.exists(image_path):
        img = preprocessor.load_and_standardize(image_path)
        binary_mask, _ = preprocessor.preprocess_for_segmentation(img)
        features = extractor.extract_features(binary_mask)
    else:
        # Fallback simulation
        features = {
            "total_roof_area_m2": 120.0,
            "usable_solar_area_m2": 90.0,
            "aspect_ratio": 1.45,
            "orientation_angle_deg": 182.0,
            "contour_count": 1,
            "perimeter_m": 44.5
        }
    
    predictions = estimator.estimate(
        usable_area_m2=features["usable_solar_area_m2"],
        annual_irradiance=irradiance,
        roof_tilt=tilt
    )
    
    result = {
        "status": "success",
        "geometric_features": features,
        "energy_predictions": predictions
    }
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar Rooftop Capacity Prediction Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to rooftop aerial image")
    parser.add_argument("--irradiance", type=float, default=1150.0, help="Annual solar irradiance in kWh/m2 (Bavaria avg ~1150)")
    parser.add_argument("--tilt", type=float, default=30.0, help="Roof tilt angle in degrees")
    args = parser.parse_args()

    report = run_pipeline(args.image, args.irradiance, args.tilt)
    print(json.dumps(report, indent=2))
