# ☀️ Solar Rooftop Capacity Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-success)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end **Computer Vision & Machine Learning** pipeline that segments building rooftops from satellite/aerial imagery, extracts geometric and architectural features, and predicts photovoltaic capacity (kWp) alongside annual energy generation (MWh/year).

---

## 📌 Architecture & Workflow

```text
[Aerial / Satellite Image]
          │
          ▼
┌────────────────────────────────────────┐
│  1. OpenCV Preprocessing & Filter      │  -> Grayscale, Gaussian Blur, Canny Edge Detection
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  2. Morphological Contour Segmentation │  -> Roof Boundary Delineation, Usable Area Estimation
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  3. Feature Extraction Module          │  -> Usable Area (m²), Aspect Ratio, Orientation Angle
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  4. Scikit-Learn Regression Models     │  -> Random Forest Regressor & Gradient Boosting
└────────────────────────────────────────┘
          │
          ▼
[Output: Installed Capacity (kWp), Annual MWh Generation, CO2 Offset]
```

---

## 🚀 Key Features

* **OpenCV Segmentation**: Handles high-contrast reflection, shadow interference, and varying roof geometries using adaptive thresholding and morphological closing.
* **Metric Surface Extraction**: Converts pixel space into real-world square meters ($m^2$) using calibrated spatial resolutions.
* **Physics-Informed ML Regression**: Combines historical regional solar irradiance curves (e.g. Central Europe/Germany: ~1100–1250 $kWh/m^2$) with tilt adjustments.
* **Automated Environmental ROI**: Estimates annual clean power output and metric tons of $CO_2$ offset.

---

## 🛠️ Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Himanshu-Sheta/solar-rooftop-capacity-prediction.git
cd solar-rooftop-capacity-prediction
pip install -r requirements.txt
```

### 2. Run Inference
```bash
python -m src.pipeline --irradiance 1150 --tilt 30
```

Sample JSON Output:
```json
{
  "status": "success",
  "geometric_features": {
    "total_roof_area_m2": 120.0,
    "usable_solar_area_m2": 90.0,
    "aspect_ratio": 1.45,
    "orientation_angle_deg": 182.0,
    "contour_count": 1,
    "perimeter_m": 44.5
  },
  "energy_predictions": {
    "installed_capacity_kwp": 17.31,
    "predicted_annual_generation_mwh": 18.25,
    "annual_co2_savings_tons": 7.3,
    "assumed_irradiance_kwh_m2": 1150.0,
    "assumed_tilt_degrees": 30.0
  }
}
```

### 3. Run Unit Tests
```bash
pytest tests/
```

---

## 👨‍💻 Author
**Himanshu Sheta**  
M.Sc. Applied Computer Science — Technische Hochschule Deggendorf  
LinkedIn: [linkedin.com/in/himanshu-sheta](https://www.linkedin.com/in/himanshu-sheta)
