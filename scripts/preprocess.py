"""
PREPROCESSING SCRIPT
====================
This script cleans your raw sales data and prepares it for model training.

Key steps:
1. Load the CSV
2. Parse dates
3. DROP blank/empty rows (your CSV has 9 blank rows with no date)
4. Fill missing dates (create a complete date index)
5. Mark out-of-stock (zero) values as NaN
6. Interpolate NaN values (estimate what they might have been)
7. Save the cleaned dataset
"""

import pandas as pd
import numpy as np


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw sales CSV and perform all cleaning steps.
    Returns a clean DataFrame ready for feature engineering.
    """

    # ─── STEP 1: LOAD THE DATA ───────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows, columns: {df.columns.tolist()}")

    # ─── STEP 2: PARSE DATES ─────────────────────────────────────────────────
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # invalid → NaT

    # ─── STEP 3: DROP BLANK / INVALID ROWS ───────────────────────────────────
    # Your CSV has 9 completely blank rows (NaT date, all NaN values).
    # These are not real records — drop them before doing anything else.
    blank_count = df['date'].isna().sum()
    if blank_count > 0:
        print(f"  Dropping {blank_count} blank/invalid rows (no date value)")
        df = df.dropna(subset=['date'])

    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Rows after dropping blanks: {len(df)}")

    # ─── STEP 4: CHECK FOR DUPLICATE DATES ───────────────────────────────────
    dup_count = df.duplicated(subset='date').sum()
    if dup_count > 0:
        print(f"  Found {dup_count} duplicate dates — keeping last entry per date")
        df = df.drop_duplicates(subset='date', keep='last')
        df = df.sort_values('date').reset_index(drop=True)

    # ─── STEP 5: FILL MISSING DATES ──────────────────────────────────────────
    # Create a complete daily date range with no gaps.
    # If a date is missing in your CSV, we add it with NaN values.
    full_date_range = pd.date_range(
        start=df['date'].min(),
        end=df['date'].max(),
        freq='D'
    )

    df = df.set_index('date').reindex(full_date_range).rename_axis('date').reset_index()
    print(f"  After filling date gaps: {len(df)} rows")

    # ─── STEP 6: MARK OUT-OF-STOCK VALUES AS NaN ─────────────────────────────
    # CRITICAL: zeros mean "out of stock" not "zero sales".
    # Replace them with NaN so interpolation can estimate real values.
    fuel_columns = ['petrol_sales', 'super_petrol_sales', 'diesel_sales', 'super_diesel_sales']

    for col in fuel_columns:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"  {col}: replacing {zero_count} zero (OOS) values with NaN")
            df[col] = df[col].replace(0, np.nan)

    # ─── STEP 7: INTERPOLATE NaN VALUES ──────────────────────────────────────
    # Estimate what sales would have been on OOS/missing days using
    # time-based interpolation (smoothly fills between known values).
    df = df.set_index('date')

    for col in fuel_columns:
        nan_before = df[col].isna().sum()
        df[col] = df[col].interpolate(method='time')
        df[col] = df[col].ffill()   # fill any remaining NaN at the end
        df[col] = df[col].bfill()   # fill any remaining NaN at the start
        nan_after = df[col].isna().sum()
        print(f"  {col}: filled {nan_before - nan_after} NaN values")

    df = df.reset_index()

    # ─── STEP 8: FINAL VALIDATION ────────────────────────────────────────────
    print("\nFinal dataset info:")
    print(f"  Rows: {len(df)}")
    print(f"  NaN values remaining:  {df[fuel_columns].isna().sum().sum()}")
    print(f"  Zero values remaining: {(df[fuel_columns] == 0).sum().sum()}")
    print(f"  Sample:\n{df.head(3).to_string()}")

    return df


def save_clean_data(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"\n✅ Clean data saved to: {output_path}")


# ─── MAIN EXECUTION ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    raw_path   = "data/sales_data.csv"
    clean_path = "data/sales_data_clean.csv"

    df = load_and_clean_data(raw_path)
    save_clean_data(df, clean_path)
