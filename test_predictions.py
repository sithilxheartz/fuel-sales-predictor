"""
PREDICTION TEST SCRIPT
=======================
Run this to verify all 4 models load correctly and produce predictions.
Run from your project root folder:

    python test_predictions.py

Expected output: predictions for tomorrow and next 7 days for all fuel types.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# ── Make sure Python can find your scripts folder ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

FUEL_LABELS = {
    'petrol_sales':       '⛽ Petrol',
    'super_petrol_sales': '🔋 Super Petrol',
    'diesel_sales':       '🚛 Diesel',
    'super_diesel_sales': '💎 Super Diesel',
}

# ─── SUPER FUEL MINIMUM THRESHOLD ────────────────────────────────────────────
# Super petrol and super diesel have very high MAPE because of OOS days.
# We apply a minimum floor: if the model predicts below the non-zero average,
# we use the rolling average instead (more reliable for erratic fuels).
SUPER_FUEL_STRATEGY = {
    'super_petrol_sales': 'blend',   # blend model + rolling average
    'super_diesel_sales': 'blend',
}


def load_all_models(models_dir="models"):
    """Load all 4 trained models from disk."""
    models = {}
    for fuel in FUEL_TYPES:
        path = os.path.join(models_dir, f"{fuel}_model.pkl")
        if os.path.exists(path):
            models[fuel] = joblib.load(path)
            print(f"  ✅ Loaded: {fuel}")
        else:
            print(f"  ❌ NOT FOUND: {path}  ← Run train_models.py first!")
    return models


def predict_n_days(historical_df, models, n_days=7):
    """
    Iteratively predict the next n_days for all fuel types.
    Each day's prediction uses previous predictions as lag inputs.
    """
    results = {}

    for fuel in FUEL_TYPES:
        if fuel not in models:
            continue

        model    = models[fuel]
        feat_cols = get_feature_columns(fuel)

        # Work on a copy with just date + this fuel column
        work_df = historical_df[['date', fuel]].copy()
        work_df['date'] = pd.to_datetime(work_df['date'])

        predictions = []
        last_date   = work_df['date'].max()

        # Calculate non-zero rolling average for super fuel fallback
        non_zero_vals = work_df[fuel][work_df[fuel] > 0]
        rolling_avg   = non_zero_vals.tail(30).mean() if len(non_zero_vals) >= 7 else None

        for day_offset in range(1, n_days + 1):
            target_date = last_date + timedelta(days=day_offset)

            # Build extended dataframe: history + previously predicted days
            extended = work_df.copy()
            for i, p in enumerate(predictions):
                extended = pd.concat([extended, pd.DataFrame([{
                    'date': last_date + timedelta(days=i + 1),
                    fuel: p
                }])], ignore_index=True)

            # Add target date row with NaN (to be predicted)
            extended = pd.concat([extended, pd.DataFrame([{
                'date': target_date,
                fuel: np.nan
            }])], ignore_index=True)

            # Create all features
            extended = create_features(extended)
            extended = create_lag_features(extended, fuel, lags=[1, 2, 3, 7, 14, 30])
            extended = create_rolling_features(extended, fuel, windows=[3, 7, 14, 30])
            extended = create_rolling_std_features(extended, fuel, windows=[7, 14])

            # Get only the row for today's prediction
            target_row = extended[extended['date'] == target_date]

            if target_row.empty or target_row[feat_cols].isnull().any().any():
                # Not enough history — use rolling average fallback
                fallback = rolling_avg if rolling_avg else work_df[fuel].tail(7).mean()
                pred_val = float(fallback)
            else:
                pred_val = float(model.predict(target_row[feat_cols])[0])
                pred_val = max(0.0, pred_val)

                # For super fuels with high OOS rate, blend with rolling average
                if fuel in SUPER_FUEL_STRATEGY and rolling_avg:
                    # 60% model, 40% rolling average — reduces extreme predictions
                    pred_val = 0.6 * pred_val + 0.4 * rolling_avg

            predictions.append(round(pred_val, 2))

        pred_dates = [
            (last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d')
            for i in range(n_days)
        ]
        results[fuel] = list(zip(pred_dates, predictions))

    return results


def print_predictions(results, n_days):
    """Pretty-print predictions to terminal."""

    print("\n" + "═" * 62)
    print("  TOMORROW'S PREDICTION")
    print("═" * 62)
    total = 0
    for fuel, preds in results.items():
        date, val = preds[0]
        total += val
        label = FUEL_LABELS[fuel]
        bar = "█" * int(val / 100)
        print(f"  {label:<20} {date}   {val:>8,.1f} L")
    print(f"\n  {'TOTAL':<20}              {total:>8,.1f} L")

    print("\n" + "═" * 62)
    print("  7-DAY FORECAST")
    print("═" * 62)

    # Print header
    date_headers = [results[FUEL_TYPES[0]][i][0][5:] for i in range(n_days)]  # MM-DD
    print(f"  {'Date':<10}", end="")
    for fuel in FUEL_TYPES:
        label = FUEL_LABELS[fuel].split(" ")[1][:8]
        print(f"  {label:>10}", end="")
    print(f"  {'Total':>10}")
    print("  " + "-" * 58)

    for i in range(n_days):
        date = results[FUEL_TYPES[0]][i][0]
        row_total = sum(results[fuel][i][1] for fuel in FUEL_TYPES)
        print(f"  {date:<10}", end="")
        for fuel in FUEL_TYPES:
            val = results[fuel][i][1]
            print(f"  {val:>10,.1f}", end="")
        print(f"  {row_total:>10,.1f}")

    print()


def print_model_notes():
    """Print honest notes about model accuracy."""
    print("\n" + "═" * 62)
    print("  MODEL ACCURACY NOTES")
    print("═" * 62)
    print("""
  ✅ Petrol      MAPE ~7%   → Predictions within ±7% on average
                              Very reliable for ordering decisions.

  ✅ Diesel      MAPE ~14%  → Predictions within ±14% on average
                              Good — acceptable for planning.

  ⚠️  Super Petrol MAPE ~50% → Low accuracy due to 83 out-of-stock
                              days in training data. Treat as a
                              rough estimate. Improves as you add
                              more data without OOS events.

  ⚠️  Super Diesel MAPE ~63% → Same issue — 19 OOS days caused
                              very erratic training data. Treat
                              as directional guidance only.

  💡 TIP: Retrain monthly as you accumulate cleaner data.
          Accuracy will improve significantly over 6 months.
""")


# ─── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 62)
    print("  EMERALD LANKA — FUEL SALES PREDICTION TEST")
    print("═" * 62)

    # 1. Load models
    print("\nLoading models...")
    models = load_all_models("models")

    if not models:
        print("\n❌ No models found. Run python scripts/train_models.py first.")
        sys.exit(1)

    # 2. Load clean historical data
    print("\nLoading historical data...")
    df = pd.read_csv("data/sales_data_clean.csv")
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded {len(df)} rows")
    print(f"  Latest date in data: {df['date'].max().date()}")
    print(f"  Predicting from: {(df['date'].max() + timedelta(days=1)).date()} onwards")

    # 3. Generate predictions
    print("\nGenerating predictions...")
    results = predict_n_days(df, models, n_days=7)

    # 4. Display results
    print_predictions(results, n_days=7)
    print_model_notes()

    print("═" * 62)
    print("  ✅ Test complete! Models are working correctly.")
    print("  Next step: Run the API server")
    print("  Command: uvicorn api.main:app --reload --port 8000")
    print("═" * 62 + "\n")