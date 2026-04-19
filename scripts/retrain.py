"""
RETRAIN.PY — EMERALD LANKA
===========================
Fetches new fuel sales data from Firebase Firestore and retrains all 4 models.

Your Firestore structure (confirmed from screenshot):
  Collection : fuelSaleHistory
  Document ID: "2026-03-28"  (date string)
  Fields:
    92PetrolSale   : 500    → petrol_sales
    95PetrolSale   : 500    → super_petrol_sales
    dieselSale     : 200    → diesel_sales
    superDieselSale: 308    → super_diesel_sales
    date           : Timestamp

Run manually:
    python scripts/retrain.py

Or triggered via API:
    POST http://localhost:8000/retrain
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── FIREBASE INIT ────────────────────────────────────────────────────────────

_firebase_app = None

def _init_firebase():
    """
    Initialize Firebase. Tries two methods:
      1. Environment variable FIREBASE_CREDENTIALS (used on Render cloud)
      2. Local file firebase/serviceAccountKey.json (used on your PC)
    """
    global _firebase_app
    if _firebase_app is not None:
        return  # already initialized

    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        _firebase_app = firebase_admin.get_app()
        return

    # Method 1: environment variable (Render deployment)
    env_creds = os.environ.get('FIREBASE_CREDENTIALS')
    if env_creds:
        print("  Using Firebase credentials from environment variable")
        cred_dict = json.loads(env_creds)
        cred = credentials.Certificate(cred_dict)
    else:
        # Method 2: local file (your PC)
        key_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'firebase', 'serviceAccountKey.json'
        )
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"\n❌ Firebase key not found at: {key_path}\n"
                "Please download it from Firebase Console:\n"
                "  Project Settings → Service Accounts → Generate new private key\n"
                "  Save as: firebase/serviceAccountKey.json"
            )
        print(f"  Using Firebase credentials from: {key_path}")
        cred = credentials.Certificate(key_path)

    import firebase_admin
    _firebase_app = firebase_admin.initialize_app(cred)
    print("  ✅ Firebase connected")


# ─── FETCH FROM FIRESTORE ─────────────────────────────────────────────────────

def fetch_from_firestore() -> pd.DataFrame:
    """
    Fetches all documents from the fuelSaleHistory collection.

    Maps Firestore fields to our training column names:
      92PetrolSale    → petrol_sales
      95PetrolSale    → super_petrol_sales
      dieselSale      → diesel_sales
      superDieselSale → super_diesel_sales
    """
    _init_firebase()
    from firebase_admin import firestore

    db = firestore.client()

    print("  Fetching documents from fuelSaleHistory...")
    docs = db.collection('fuelSaleHistory').stream()

    records = []
    skipped = 0

    for doc in docs:
        data = doc.to_dict()

        # ── Parse date ────────────────────────────────────────────────────
        # Document ID is the date string e.g. "2026-03-28"
        # We use the doc ID as the primary date source (more reliable)
        doc_id = doc.id  # e.g. "2026-03-28"

        try:
            date = pd.to_datetime(doc_id)
        except Exception:
            # Fallback: try the 'date' field (Firestore Timestamp)
            raw_date = data.get('date')
            if raw_date is None:
                skipped += 1
                continue
            # Firestore Timestamps have a .date() method or similar
            try:
                date = pd.to_datetime(raw_date.isoformat()
                                      if hasattr(raw_date, 'isoformat')
                                      else str(raw_date))
            except Exception:
                skipped += 1
                continue

        # ── Map field names ───────────────────────────────────────────────
        def safe_float(key, fallback=np.nan):
            val = data.get(key)
            if val is None:
                return fallback
            try:
                return float(val)
            except (ValueError, TypeError):
                return fallback

        records.append({
            'date':               date,
            'petrol_sales':       safe_float('92PetrolSale'),
            'super_petrol_sales': safe_float('95PetrolSale'),
            'diesel_sales':       safe_float('dieselSale'),
            'super_diesel_sales': safe_float('superDieselSale'),
        })

    if skipped > 0:
        print(f"  ⚠️  Skipped {skipped} documents with unparseable dates")

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No records fetched from Firestore. Check collection name and permissions.")

    df = df.sort_values('date').reset_index(drop=True)

    print(f"  ✅ Fetched {len(df)} records")
    print(f"     Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"     Sample:\n{df.head(3).to_string()}")

    return df


# ─── MERGE WITH HISTORICAL CSV ────────────────────────────────────────────────

def merge_with_historical(firebase_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Firebase data with the existing CSV data.

    Strategy:
    - Start with original CSV data (April 2025 → January 2026)
    - Add Firebase records (new data from Firebase)
    - For any date that exists in BOTH, Firebase data wins
      (it's more up-to-date / corrected)
    - Sort by date, remove duplicates
    """
    original_path = os.path.join('data', 'sales_data.csv')
    clean_path    = os.path.join('data', 'sales_data_clean.csv')

    # Load the best available historical data
    if os.path.exists(clean_path):
        print(f"  Loading existing clean data from: {clean_path}")
        existing = pd.read_csv(clean_path)
    elif os.path.exists(original_path):
        print(f"  Loading original CSV from: {original_path}")
        existing = pd.read_csv(original_path)
    else:
        print("  ⚠️  No existing CSV found — using Firebase data only")
        existing = pd.DataFrame(columns=[
            'date', 'petrol_sales', 'super_petrol_sales',
            'diesel_sales', 'super_diesel_sales'
        ])

    existing['date'] = pd.to_datetime(existing['date'], errors='coerce')
    existing = existing.dropna(subset=['date'])

    firebase_df['date'] = pd.to_datetime(firebase_df['date'])

    # Show what's new
    existing_dates  = set(existing['date'].dt.strftime('%Y-%m-%d'))
    firebase_dates  = set(firebase_df['date'].dt.strftime('%Y-%m-%d'))
    new_dates       = firebase_dates - existing_dates
    updated_dates   = firebase_dates & existing_dates

    print(f"\n  Merge summary:")
    print(f"    Existing records : {len(existing)}")
    print(f"    Firebase records : {len(firebase_df)}")
    print(f"    New dates        : {len(new_dates)}")
    print(f"    Updated dates    : {len(updated_dates)}")

    # Concatenate — Firebase last so it overwrites on dedup
    combined = pd.concat([existing, firebase_df], ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'])

    # Keep Firebase version for any overlapping dates (keep='last')
    combined = combined.drop_duplicates(subset='date', keep='last')
    combined = combined.sort_values('date').reset_index(drop=True)

    print(f"    Total after merge: {len(combined)} records")
    print(f"    Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")

    return combined


# ─── SAVE MERGED DATA ─────────────────────────────────────────────────────────

def save_merged(df: pd.DataFrame, path: str = 'data/sales_data_merged.csv'):
    os.makedirs('data', exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\n  ✅ Merged data saved → {path}")
    return path


# ─── FULL RETRAIN PIPELINE ────────────────────────────────────────────────────

def full_retrain_pipeline():
    """
    Complete pipeline called by the API's /retrain endpoint.

    Steps:
      1. Fetch new data from Firestore (fuelSaleHistory)
      2. Merge with historical CSV data
      3. Save merged CSV
      4. Preprocess (handle OOS zeros, fill date gaps, interpolate)
      5. Retrain all 4 XGBoost models
      6. Save updated models to models/
    """
    print("\n" + "=" * 60)
    print("  RETRAIN PIPELINE — EMERALD LANKA")
    print("=" * 60)
    started_at = datetime.now()

    # ── Step 1: Fetch from Firebase ───────────────────────────────────────
    print("\n[1/5] Fetching data from Firebase fuelSaleHistory...")
    firebase_df = fetch_from_firestore()

    # ── Step 2: Merge ─────────────────────────────────────────────────────
    print("\n[2/5] Merging with historical CSV data...")
    merged_df = merge_with_historical(firebase_df)

    # ── Step 3: Save merged CSV ───────────────────────────────────────────
    print("\n[3/5] Saving merged dataset...")
    merged_path = save_merged(merged_df)

    # ── Step 4: Preprocess ────────────────────────────────────────────────
    print("\n[4/5] Preprocessing (handling OOS zeros, filling gaps)...")
    from scripts.preprocess import load_and_clean_data, save_clean_data
    clean_df = load_and_clean_data(merged_path)
    save_clean_data(clean_df, 'data/sales_data_clean.csv')

    # ── Step 5: Retrain all 4 models ──────────────────────────────────────
    print("\n[5/5] Retraining all 4 XGBoost models...")
    from scripts.train_models import train_all_models
    metrics = train_all_models(
        clean_data_path='data/sales_data_clean.csv',
        models_dir='models'
    )

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - started_at).seconds
    print("\n" + "=" * 60)
    print(f"  ✅ RETRAIN COMPLETE in {elapsed}s")
    print("=" * 60)
    print("\n  New model accuracy:")
    for fuel, m in metrics.items():
        print(f"    {fuel:<30} MAPE: {m['mape']:.2f}%")

    return metrics


# ─── OPTIONAL: PUSH PREDICTIONS BACK TO FIREBASE ─────────────────────────────

def push_predictions_to_firestore(predictions: dict):
    """
    Optionally stores the latest predictions in Firestore so the Flutter app
    can read predictions offline (without calling the API).

    Writes to collection: fuelPredictions
    Document ID: "latest"
    """
    _init_firebase()
    from firebase_admin import firestore

    db = firestore.client()
    doc = {
        'generatedAt': datetime.now().isoformat(),
        'tomorrow': predictions.get('tomorrow', {}),
        'sevenDays': predictions.get('seven_days', {}),
    }
    db.collection('fuelPredictions').document('latest').set(doc)
    print("✅ Predictions pushed to Firestore → fuelPredictions/latest")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nStarting manual retrain...")
    print("This will:")
    print("  1. Connect to Firebase fuelSaleHistory")
    print("  2. Merge with your local CSV data")
    print("  3. Preprocess & retrain all 4 models")
    print()

    try:
        metrics = full_retrain_pipeline()
        print("\n✅ Done! Your models are now updated with the latest Firebase data.")
        print("   Restart the API server to use the new models:")
        print("   uvicorn api.main:app --reload --port 8000")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"\n❌ Retrain failed: {e}")
        raise