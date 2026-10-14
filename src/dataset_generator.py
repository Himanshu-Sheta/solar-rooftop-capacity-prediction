import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class GermanSolarDatasetGenerator:
    """
    Generates synthetic calibrated solar datasets based on German meteorological indices
    (DWD - Deutscher Wetterdienst reference irradiance for Bavaria, Baden-Württemberg, etc.).
    """
    REGIONAL_IRRADIANCE_KWH_M2 = {
        "Deggendorf (Bavaria)": 1165.0,
        "Munich (Bavaria)": 1180.0,
        "Regensburg (Bavaria)": 1150.0,
        "Nuremberg (Bavaria)": 1130.0,
        "Stuttgart (Baden-Württemberg)": 1145.0,
        "Frankfurt (Hesse)": 1090.0,
        "Berlin (Brandenburg)": 1040.0
    }

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_rooftop_cohort(self, sample_size: int = 1200) -> pd.DataFrame:
        regions = list(self.REGIONAL_IRRADIANCE_KWH_M2.keys())
        assigned_regions = np.random.choice(regions, size=sample_size)
        base_irradiance = np.array([self.REGIONAL_IRRADIANCE_KWH_M2[r] for r in assigned_regions])
        
        # Add micro-climate variance (+/- 3%)
        actual_irradiance = base_irradiance * np.random.uniform(0.97, 1.03, size=sample_size)
        
        # Rooftop physical distributions
        total_roof_area = np.random.lognormal(mean=4.7, sigma=0.5, size=sample_size).clip(30.0, 450.0)
        usable_fraction = np.random.uniform(0.65, 0.82, size=sample_size)
        usable_area = total_roof_area * usable_fraction
        
        # Roof pitch (degrees) - peak distribution around 30-38 degrees in Germany
        tilt_degrees = np.random.normal(loc=34.0, scale=6.0, size=sample_size).clip(15.0, 50.0)
        
        # Azimuth angle (0=South, -90=East, +90=West)
        azimuth_deg = np.random.normal(loc=0.0, scale=35.0, size=sample_size).clip(-80.0, 80.0)
        
        # Orientation factor calculation
        tilt_loss = np.cos(np.radians(tilt_degrees - 32.0))
        azimuth_loss = np.cos(np.radians(azimuth_deg * 0.75))
        efficiency_factor = (tilt_loss * azimuth_loss).clip(0.70, 1.0)
        
        # Monocrystalline silicon PV module efficiency ~ 20.8%
        pv_efficiency = 0.208
        installed_capacity_kwp = (usable_area * pv_efficiency) / 1.0
        
        # System Performance Ratio (inverter losses, temperature coefficient ~ 0.83)
        performance_ratio = 0.83
        annual_generation_mwh = (installed_capacity_kwp * actual_irradiance * efficiency_factor * performance_ratio) / 1000.0
        
        # Environmental impact (405g CO2 saved per kWh solar generated in Germany)
        co2_savings_tons_yr = (annual_generation_mwh * 1000.0 * 0.405) / 1000.0
        
        df = pd.DataFrame({
            "region": assigned_regions,
            "total_roof_area_m2": np.round(total_roof_area, 2),
            "usable_area_m2": np.round(usable_area, 2),
            "roof_tilt_deg": np.round(tilt_degrees, 2),
            "azimuth_deg": np.round(azimuth_deg, 2),
            "annual_irradiance_kwh": np.round(actual_irradiance, 2),
            "capacity_kwp": np.round(installed_capacity_kwp, 2),
            "annual_generation_mwh": np.round(annual_generation_mwh, 2),
            "co2_savings_tons_yr": np.round(co2_savings_tons_yr, 2)
        })
        return df

if __name__ == "__main__":
    gen = GermanSolarDatasetGenerator()
    data = gen.generate_rooftop_cohort(200)
    print(f"Generated {len(data)} realistic rooftop solar profiles across German regions.")
