"""
PREDICT.PY — EMERALD LANKA FUEL PREDICTION
===========================================
Loads trained models and generates predictions.
Used by the FastAPI server (api/main.py).

Prediction types:
  - Tomorrow (next 1 day)
  - Next 7 days

Super fuel strategy:
  Super petrol (MAPE ~50%) and super diesel (MAPE ~63%) have poor accuracy
  because of heavy out-of-stock periods in training data.
  We blend the model output with a rolling average to produce more stable,
  usable predictions for these fuels.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.feature_engineering import (
    create_features, create_lag_features,
    create_rolling_features, create_rolling_std_features,
    get_feature_columns
)

FUEL_TYPES = [
    'petrol_sales',
    'super_petrol_sales',
    'diesel_sales',
    'super_diesel_sales',
]

# For super fuels, blend model prediction with rolling average.
# Ratio: model_weight * model_pred + (1 - model_weight) * rolling_avg
BLEND_CONFIG = {
    'super_petrol_sales': 0.55,  # 55% model, 45% rolling average
    'super_diesel_sales': 0.45,  # 45% model, 55% rolling average (more erratic)
}


def load_models(models_dir: str = "models") -> dict:
    """Load all 4 trained models from disk into memory."""
    models = {}
    for fuel in FUEL_TYPES:
        path = os.path.join(models_dir, f"{fuel}_model.pkl")
        if os.path.exists(path):
            models[fuel] = joblib.load(path)
        else:
            print(f"  ⚠️  Model not found: {path}")
    return models


def _predict_single_fuel(
    historical_df: pd.DataFrame,
    fuel: str,
    model,
    n_days: int
) -> list:
    """
    Core prediction loop for a single fuel type.
    Returns a list of (date_string, predicted_value) tuples.

    Uses iterative/recursive forecasting:
    - Day 1: use real history as lag inputs
    - Day 2: use real history + Day 1 prediction as lag inputs
    - Day N: use real history + Day 1..N-1 predictions
    """
    feat_cols = get_feature_columns(fuel)

    work_df = historical_df[['date', fuel]].copy()
    work_df['date'] = pd.to_datetime(work_df['date'])
    work_df = work_df.sort_values('date').reset_index(drop=True)

    last_date = work_df['date'].max()

    # Calculate non-zero rolling average for fallback / blending
    non_zero = work_df[fuel][work_df[fuel] > 0]
    rolling_avg_30 = float(non_zero.tail(30).mean()) if len(non_zero) >= 5 else None
    rolling_avg_7  = float(non_zero.tail(7).mean())  if len(non_zero) >= 3 else None

    predictions = []

    for day_offset in range(1, n_days + 1):
        target_date = last_date + timedelta(days=day_offset)

        # Build extended dataframe including all previously predicted values
        extended = work_df.copy()
        for i, (_, pred_val) in enumerate(predictions):
            extended = pd.concat([
                extended,
                pd.DataFrame([{
                    'date': last_date + timedelta(days=i + 1),
                    fuel: pred_val
                }])
            ], ignore_index=True)

        # Add target date with NaN target
        extended = pd.concat([
            extended,
            pd.DataFrame([{'date': target_date, fuel: np.nan}])
        ], ignore_index=True)

        # Create features on the extended dataframe
        extended = create_features(extended)
        extended = create_lag_features(extended, fuel, lags=[1, 2, 3, 7, 14, 30])
        extended = create_rolling_features(extended, fuel, windows=[3, 7, 14, 30])
        extended = create_rolling_std_features(extended, fuel, windows=[7, 14])

        # Select only the target date row
        row = extended[extended['date'] == target_date]

        # ── PREDICTION ────────────────────────────────────────────────────────
        fallback = rolling_avg_7 or rolling_avg_30 or work_df[fuel].tail(7).mean()

        if row.empty or row[feat_cols].isnull().any().any():
            # Not enough historical data to compute all lag features
            pred_val = fallback
        else:
            pred_val = float(model.predict(row[feat_cols])[0])
            pred_val = max(0.0, pred_val)

            # For super fuels: blend model output with rolling average
            if fuel in BLEND_CONFIG and rolling_avg_30:
                w = BLEND_CONFIG[fuel]
                pred_val = w * pred_val + (1 - w) * rolling_avg_30

        predictions.append((
            target_date.strftime('%Y-%m-%d'),
            round(pred_val, 2)
        ))

    return predictions


def predict_all_fuels(
    historical_df: pd.DataFrame,
    models: dict,
    n_days: int = 7
) -> dict:
    """
    Predict the next n_days for all 4 fuel types.

    Returns:
    {
      'petrol_sales': [('2026-02-01', 2450.5), ('2026-02-02', 2380.1), ...],
      'super_petrol_sales': [...],
      ...
    }
    """
    results = {}
    for fuel in FUEL_TYPES:
        if fuel not in models:
            continue
        results[fuel] = _predict_single_fuel(
            historical_df, fuel, models[fuel], n_days
        )
    return results


def get_tomorrow_prediction(historical_df: pd.DataFrame, models: dict) -> dict:
    """
    Returns tomorrow's prediction for each fuel type.

    Returns:
    {
      'petrol_sales':       {'date': '2026-02-01', 'predicted_litres': 2450.5},
      'super_petrol_sales': {'date': '2026-02-01', 'predicted_litres': 72.3},
      'diesel_sales':       {'date': '2026-02-01', 'predicted_litres': 3100.8},
      'super_diesel_sales': {'date': '2026-02-01', 'predicted_litres': 115.2},
    }
    """
    all_preds = predict_all_fuels(historical_df, models, n_days=1)
    return {
        fuel: {
            'date': preds[0][0],
            'predicted_litres': preds[0][1]
        }
        for fuel, preds in all_preds.items()
    }


def get_7day_prediction(historical_df: pd.DataFrame, models: dict) -> dict:
    """
    Returns 7-day predictions for each fuel type.

    Returns:
    {
      'petrol_sales': [
        {'date': '2026-02-01', 'predicted_litres': 2450.5},
        {'date': '2026-02-02', 'predicted_litres': 2380.1},
        ...
      ],
      ...
    }
    """
    all_preds = predict_all_fuels(historical_df, models, n_days=7)
    return {
        fuel: [
            {'date': date, 'predicted_litres': val}
            for date, val in preds
        ]
        for fuel, preds in all_preds.items()
    }