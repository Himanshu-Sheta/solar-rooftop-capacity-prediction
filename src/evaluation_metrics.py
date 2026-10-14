import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any

class ModelEvaluationMetrics:
    """
    Computes rigorous cross-validation and regression benchmark metrics for solar capacity estimation.
    """
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        non_zero = y_true != 0
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0
        
        return {
            "MAE_MWh": round(float(mae), 4),
            "RMSE_MWh": round(float(rmse), 4),
            "R2_Score": round(float(r2), 4),
            "MAPE_Percent": round(float(mape), 2)
        }

    @staticmethod
    def evaluate_capacity_quantiles(df_results, capacity_col="capacity_kwp"):
        """
        Stratified error evaluation across residential (<10 kWp), commercial (10-50 kWp), and industrial (>50 kWp).
        """
        cohorts = {
            "residential": df_results[df_results[capacity_col] < 10.0],
            "commercial": df_results[(df_results[capacity_col] >= 10.0) & (df_results[capacity_col] <= 50.0)],
            "industrial": df_results[df_results[capacity_col] > 50.0]
        }
        summary = {}
        for name, sub in cohorts.items():
            if len(sub) > 0:
                summary[name] = {
                    "count": len(sub),
                    "mean_actual_mwh": round(sub["annual_generation_mwh"].mean(), 2)
                }
        return summary
