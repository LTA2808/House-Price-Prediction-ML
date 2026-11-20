"""Recompute the recorded metrics from the published test predictions."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = pd.read_csv(ROOT / "results" / "test_predictions.csv")
SUMMARY = pd.read_csv(ROOT / "results" / "model_comparison.csv")
y_true = PRED["Actual_USD"].to_numpy(dtype=float)
models = {
    "Linear Regression": "LinearRegression_Pred_USD",
    "Random Forest": "RandomForest_Pred_USD",
}
for model_name, column in models.items():
    y_pred = PRED[column].to_numpy(dtype=float)
    difference = y_true - y_pred
    mae = float(np.abs(difference).mean())
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    r2 = float(1 - np.sum(difference ** 2) / np.sum((y_true - y_true.mean()) ** 2))
    recorded = SUMMARY.loc[SUMMARY["Model"] == model_name].iloc[0]
    assert np.isclose(mae, recorded["MAE_USD"], rtol=1e-10, atol=1e-7)
    assert np.isclose(rmse, recorded["RMSE_USD"], rtol=1e-10, atol=1e-7)
    assert np.isclose(r2, recorded["R2"], rtol=1e-10, atol=1e-7)
    print(f"{model_name}: MAE=${mae:,.2f}, RMSE=${rmse:,.2f}, R2={r2:.4f} [PASS]")
print(f"Verified {len(PRED):,} test samples; published metrics match predictions.")
