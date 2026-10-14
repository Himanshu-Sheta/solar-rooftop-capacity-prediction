import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from typing import Dict, Any, Tuple

class SolarModelTrainer:
    """
    Automated model training and hyperparameter optimization suite
    for rooftop solar generation forecasting across varied building profiles.
    """
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.best_model = None
        self.training_history = {}

    def train_and_compare_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        candidate_models = {
            "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)
        }
        
        benchmark_results = {}
        for name, model in candidate_models.items():
            r2_scores = cross_val_score(model, X, y, cv=self.kfold, scoring="r2")
            neg_mae_scores = cross_val_score(model, X, y, cv=self.kfold, scoring="neg_mean_absolute_error")
            
            benchmark_results[name] = {
                "mean_r2": round(float(np.mean(r2_scores)), 4),
                "std_r2": round(float(np.std(r2_scores)), 4),
                "mean_mae_mwh": round(float(-np.mean(neg_mae_scores)), 4)
            }
            
        best_name = max(benchmark_results.keys(), key=lambda k: benchmark_results[k]["mean_r2"])
        self.best_model = candidate_models[best_name]
        self.best_model.fit(X, y)
        
        return {
            "best_performing_model": best_name,
            "benchmark_summary": benchmark_results,
            "training_samples": len(X)
        }
