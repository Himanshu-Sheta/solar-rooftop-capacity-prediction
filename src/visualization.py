import matplotlib.pyplot as plt
import numpy as np
import os

class SolarVisualizationSuite:
    """
    Generates visual analytics for model evaluation, feature correlations, and rooftop segmentation masks.
    """
    def __init__(self, output_dir: str = "artifacts"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "actual_vs_pred.png"):
        plt.figure(figsize=(8, 6), dpi=150)
        plt.scatter(y_true, y_pred, alpha=0.6, color="#2563EB", edgecolors="none", s=25)
        
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], "--", color="#DC2626", linewidth=1.5, label="Perfect Agreement (1:1)")
        
        plt.title("Rooftop Solar Generation: Actual vs Predicted (MWh/year)", fontsize=13, fontweight="bold")
        plt.xlabel("Simulated Physics Yield (MWh)", fontsize=11)
        plt.ylabel("ML Predicted Yield (MWh)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        
        full_path = os.path.join(self.output_dir, save_path)
        plt.savefig(full_path)
        plt.close()
        return full_path
