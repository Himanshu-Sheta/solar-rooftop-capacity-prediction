import pytest
import numpy as np
from src.feature_extractor import RooftopFeatureExtractor
from src.model import SolarCapacityEstimator
from src.pipeline import run_pipeline

def test_feature_extractor_synthetic_mask():
    extractor = RooftopFeatureExtractor(pixels_per_meter_ratio=10.0)
    # Create 100x100 square mask (10m x 10m = 100m2)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 50:150] = 255
    
    features = extractor.extract_features(mask)
    assert features["total_roof_area_m2"] == 100.0
    assert features["usable_solar_area_m2"] == 75.0
    assert features["contour_count"] == 1

def test_model_capacity_estimation():
    model = SolarCapacityEstimator()
    est = model.estimate(usable_area_m2=100.0, annual_irradiance=1100.0, roof_tilt=30.0)
    assert est["installed_capacity_kwp"] > 0.0
    assert est["predicted_annual_generation_mwh"] > 0.0
    assert est["annual_co2_savings_tons"] > 0.0

def test_full_pipeline_run():
    res = run_pipeline(image_path=None)
    assert res["status"] == "success"
    assert "geometric_features" in res
    assert "energy_predictions" in res
