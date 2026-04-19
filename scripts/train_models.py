"""
TRAIN_MODELS.PY — EMERALD LANKA (Improved)
============================================
Trains 4 XGBoost models, one per fuel type.

Key improvements over original:
  - Smarter test split: uses 20 days instead of 30 (better for small datasets)
  - TimeSeriesSplit cross-validation for more reliable MAPE estimate
  - Shows CV MAPE alongside test MAPE so you know which to trust
  - Fixed n_estimators duplicate argument bug
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.feature_engineering import prepare_features_for_fuel, get_feature_columns


FUEL_TYPES = [
    'petrol_sales',
    'super_petrol_sales',
    'diesel_sales',
    'super_diesel_sales',
]

XGBOOST_PARAMS = {
    'n_estimators':        500,
    'max_depth':           5,
    'learning_rate':       0.05,
    'subsample':           0.8,
    'colsample_bytree':    0.8,
    'random_state':        42,
    'n_jobs':              -1,
    'early_stopping_rounds': 50,
}


def train_model_for_fuel(df: pd.DataFrame, target_col: str, models_dir: str) -> dict:

    print(f"\n{'='*60}")
    print(f"  Training model for: {target_col}")
    print(f"{'='*60}")

    # ── Feature preparation ───────────────────────────────────────────────────
    df_feat      = prepare_features_for_fuel(df[['date', target_col]].copy(), target_col)
    feature_cols = get_feature_columns(target_col)
    X = df_feat[feature_cols]
    y = df_feat[target_col]

    print(f"  Dataset size: {len(X)} rows, {len(feature_cols)} features")

    # ── TimeSeriesSplit Cross-Validation ──────────────────────────────────────
    # More reliable than a single train/test split for small datasets.
    # Splits the data into 5 folds, always training on past and testing on future.
    tscv     = TimeSeriesSplit(n_splits=5)
    cv_maes  = []
    cv_mapes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Need a validation set within training for early stopping
        val_split  = int(len(X_tr) * 0.9)
        X_tr2      = X_tr.iloc[:val_split]
        y_tr2      = y_tr.iloc[:val_split]
        X_es_val   = X_tr.iloc[val_split:]
        y_es_val   = y_tr.iloc[val_split:]

        fold_model = XGBRegressor(**XGBOOST_PARAMS)
        fold_model.fit(
            X_tr2, y_tr2,
            eval_set=[(X_es_val, y_es_val)],
            verbose=False
        )

        y_pred = fold_model.predict(X_val)
        y_pred = np.maximum(0, y_pred)

        mae  = mean_absolute_error(y_val, y_pred)
        # MAPE: only on non-zero actuals to avoid divide-by-zero
        mask = y_val > 0
        mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100

        cv_maes.append(mae)
        cv_mapes.append(mape)

    cv_mae_mean  = np.mean(cv_maes)
    cv_mape_mean = np.mean(cv_mapes)

    print(f"\n  Cross-validation (5-fold TimeSeriesSplit):")
    print(f"    CV MAE  : {cv_mae_mean:.2f} L  (±{np.std(cv_maes):.2f})")
    print(f"    CV MAPE : {cv_mape_mean:.2f}%  (±{np.std(cv_mapes):.2f}%)")
    print(f"    ← This is the most reliable accuracy estimate")

    # ── Holdout test set (last 20 days) ───────────────────────────────────────
    # Used as a final sanity check — not for model selection
    test_size  = 20
    train_size = len(X) - test_size

    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    val_split = int(len(X_train) * 0.9)
    X_tr  = X_train.iloc[:val_split]
    y_tr  = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]

    holdout_model = XGBRegressor(**XGBOOST_PARAMS)
    holdout_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred    = np.maximum(0, holdout_model.predict(X_test))
    h_mae     = mean_absolute_error(y_test, y_pred)
    h_rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    mask      = y_test > 0
    h_mape    = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    print(f"\n  Holdout test (last 20 days):")
    print(f"    MAE  : {h_mae:.2f} L")
    print(f"    RMSE : {h_rmse:.2f} L")
    print(f"    MAPE : {h_mape:.2f}%")

    # ── Final model: train on ALL data ────────────────────────────────────────
    best_n = holdout_model.best_iteration + 1
    print(f"\n  Training final model on ALL {len(X)} rows (n_estimators={best_n})...")

    final_params = {k: v for k, v in XGBOOST_PARAMS.items()
                    if k not in ('early_stopping_rounds', 'n_estimators')}
    final_params['n_estimators'] = best_n

    final_model = XGBRegressor(**final_params)
    final_model.fit(X, y)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(final_model,   os.path.join(models_dir, f"{target_col}_model.pkl"))
    joblib.dump(feature_cols,  os.path.join(models_dir, f"{target_col}_features.pkl"))
    joblib.dump({
        'cv_mae':   cv_mae_mean,
        'cv_mape':  cv_mape_mean,
        'holdout_mae':  h_mae,
        'holdout_rmse': h_rmse,
        'holdout_mape': h_mape,
        'mae':  cv_mae_mean,   # used by retrain pipeline summary
        'mape': cv_mape_mean,  # ← CV MAPE is the reliable one
    }, os.path.join(models_dir, f"{target_col}_metrics.pkl"))

    print(f"  ✅ Model saved: {models_dir}/{target_col}_model.pkl")

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n  TOP 10 MOST IMPORTANT FEATURES:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:<38} {imp:.4f}")

    return {
        'cv_mae':       cv_mae_mean,
        'cv_mape':      cv_mape_mean,
        'holdout_mape': h_mape,
        'mae':          cv_mae_mean,
        'mape':         cv_mape_mean,
    }


def train_all_models(
    clean_data_path: str = "data/sales_data_clean.csv",
    models_dir:      str = "models"
) -> dict:

    print("Loading clean data...")
    df = pd.read_csv(clean_data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df)} rows  "
          f"({df['date'].min().date()} → {df['date'].max().date()})")

    all_metrics = {}
    for fuel_type in FUEL_TYPES:
        all_metrics[fuel_type] = train_model_for_fuel(df, fuel_type, models_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*60)
    print(f"\n{'Fuel Type':<30} {'CV MAPE':>10} {'Holdout':>10} {'Reliability':>12}")
    print("-"*65)

    for fuel, m in all_metrics.items():
        cv   = m['cv_mape']
        hold = m['holdout_mape']

        if fuel in ['petrol_sales', 'diesel_sales']:
            rel = "✅ Good" if cv < 15 else ("⚠️  Fair" if cv < 25 else "❌ Poor")
        else:
            rel = "✅ OK"   if cv < 60 else ("⚠️  Fair" if cv < 80 else "❌ Poor")

        print(f"{fuel:<30} {cv:>9.2f}% {hold:>9.2f}% {rel:>12}")

    print(f"\n  Note: CV MAPE is averaged over 5 time-series folds.")
    print(f"        It is more reliable than the single holdout MAPE.")
    print(f"\n✅ All 4 models trained and saved to {models_dir}/")

    return all_metrics


if __name__ == "__main__":
    train_all_models()