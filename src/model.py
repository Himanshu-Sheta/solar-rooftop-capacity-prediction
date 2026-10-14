import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from typing import Dict, Any

class SolarCapacityEstimator:
    """
    Machine Learning model to predict Solar Peak Capacity (kWp) and Annual Generation (MWh).
    """
    def __init__(self, standard_panel_efficiency: float = 0.20):
        self.panel_efficiency = standard_panel_efficiency
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._is_trained = False
        self._train_synthetic_baseline()

    def _train_synthetic_baseline(self):
        # Baseline physics-guided training
        np.random.seed(42)
        usable_area = np.random.uniform(20.0, 300.0, 500)
        irradiance = np.random.uniform(950.0, 1300.0, 500)  # kWh/m2 (Central Europe / Germany range)
        tilt = np.random.uniform(15.0, 45.0, 500)
        
        # Performance ratio PR ~ 0.82
        pr = 0.82
        kwp = (usable_area * self.panel_efficiency)
        annual_mwh = (kwp * irradiance * pr * np.cos(np.radians(tilt - 35))) / 1000.0
        
        X = np.column_stack([usable_area, irradiance, tilt])
        y = annual_mwh + np.random.normal(0, 0.2, 500)
        
        self.model.fit(X, y)
        self._is_trained = True

    def estimate(self, usable_area_m2: float, annual_irradiance: float = 1150.0, roof_tilt: float = 30.0) -> Dict[str, Any]:
        """
        Calculates peak power and predicts yearly production.
        """
        # Photovoltaic capacity: ~1 kWp requires approx 5 m2 of high-efficiency silicon panels
        installed_capacity_kwp = round(usable_area_m2 / 5.2, 2)
        
        X_infer = np.array([[usable_area_m2, annual_irradiance, roof_tilt]])
        predicted_annual_mwh = round(float(self.model.predict(X_infer)[0]), 2)
        
        # Estimate CO2 savings: ~400g CO2 per kWh in German energy mix
        annual_co2_offset_tons = round((predicted_annual_mwh * 1000 * 0.40) / 1000.0, 2)
        
        return {
            "installed_capacity_kwp": installed_capacity_kwp,
            "predicted_annual_generation_mwh": max(0.0, predicted_annual_mwh),
            "annual_co2_savings_tons": max(0.0, annual_co2_offset_tons),
            "assumed_irradiance_kwh_m2": annual_irradiance,
            "assumed_tilt_degrees": roof_tilt
        }
