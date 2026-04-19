"""
BULK FIREBASE UPLOAD — EMERALD LANKA
=====================================
Reads your CSV file and uploads all records to Firebase fuelSaleHistory.

Your CSV columns:
  date                → Document ID (e.g. "2025-02-01")
  petrol_sales        → 92PetrolSale
  super_petrol_sales  → 95PetrolSale
  diesel_sales        → dieselSale
  super_diesel_sales  → superDieselSale

Run from your project root:
    python scripts/bulk_upload_to_firebase.py

Optional — upload a different file:
    python scripts/bulk_upload_to_firebase.py data/my_other_file.csv
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Default CSV file to upload — change this if your file is elsewhere
DEFAULT_CSV = "data/sales_data_from_02_01.csv"

# Firebase collection to upload into
COLLECTION = "fuelSaleHistory"

# Firestore field names (must match what your Flutter app expects)
FIELD_MAP = {
    'petrol_sales':       '92PetrolSale',
    'super_petrol_sales': '95PetrolSale',
    'diesel_sales':       'dieselSale',
    'super_diesel_sales': 'superDieselSale',
}

# Batch size — Firestore allows max 500 writes per batch
BATCH_SIZE = 400


# ─── FIREBASE INIT ────────────────────────────────────────────────────────────

def init_firebase():
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return firebase_admin.get_app()

    env_creds = os.environ.get('FIREBASE_CREDENTIALS')
    if env_creds:
        cred = credentials.Certificate(json.loads(env_creds))
    else:
        key_path = os.path.join('firebase', 'serviceAccountKey.json')
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"\n❌ Firebase key not found: {key_path}\n"
                "Download from Firebase Console → Project Settings → Service Accounts"
            )
        cred = credentials.Certificate(key_path)

    app = firebase_admin.initialize_app(cred)
    print("  ✅ Firebase connected")
    return app


# ─── LOAD & VALIDATE CSV ──────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV, drop blank rows, validate columns and data."""

    print(f"\n  Loading: {filepath}")
    df = pd.read_csv(filepath)

    # Drop blank rows (your CSV has many NaN rows at the bottom)
    df = df.dropna(subset=['date'])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"  Rows loaded    : {len(df)}")
    print(f"  Date range     : {df['date'].min().date()} → {df['date'].max().date()}")

    # Check required columns exist
    required = ['date'] + list(FIELD_MAP.keys())
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Show zero/OOS counts
    print("\n  Out-of-stock (zero) counts per fuel:")
    for col in FIELD_MAP.keys():
        zeros = (df[col] == 0).sum()
        nans  = df[col].isna().sum()
        print(f"    {col:<25}: {zeros} zeros, {nans} NaN")

    return df


# ─── PREVIEW ─────────────────────────────────────────────────────────────────

def preview(df: pd.DataFrame):
    """Show a sample of what will be uploaded."""
    print("\n  Preview (first 3 records that will be uploaded):")
    print(f"  {'─'*60}")
    for _, row in df.head(3).iterrows():
        doc_id = row['date'].strftime('%Y-%m-%d')
        print(f"\n  Document ID: {doc_id}")
        for csv_col, fb_field in FIELD_MAP.items():
            val = row[csv_col]
            val_str = f"{val:.3f}" if not pd.isna(val) else "null"
            oos = " ← OOS (zero)" if val == 0 else ""
            print(f"    {fb_field:<20}: {val_str}{oos}")
    print(f"\n  {'─'*60}")


# ─── CHECK EXISTING DOCS ──────────────────────────────────────────────────────

def check_existing(db, dates: list) -> set:
    """
    Check which dates already exist in Firestore.
    Returns a set of date strings that already have documents.
    """
    existing = set()
    # Check in batches of 30 (Firestore 'in' query limit)
    for i in range(0, len(dates), 30):
        chunk = dates[i:i+30]
        for date_str in chunk:
            doc = db.collection(COLLECTION).document(date_str).get()
            if doc.exists:
                existing.add(date_str)
    return existing


# ─── UPLOAD ──────────────────────────────────────────────────────────────────

def upload_to_firebase(df: pd.DataFrame, overwrite: bool = True):
    """
    Uploads all rows to Firebase fuelSaleHistory using batched writes.

    Each row becomes a Firestore document:
      Document ID : "2025-02-01"  (date string)
      Fields      : 92PetrolSale, 95PetrolSale, dieselSale, superDieselSale, date

    overwrite=True  → replaces existing documents (default)
    overwrite=False → skips dates that already exist in Firestore
    """
    from firebase_admin import firestore

    db = firestore.client()

    # Prepare all records
    all_dates  = [row['date'].strftime('%Y-%m-%d') for _, row in df.iterrows()]

    # Check what already exists
    if not overwrite:
        print("\n  Checking existing Firestore documents...")
        existing = check_existing(db, all_dates)
        print(f"  Already in Firestore: {len(existing)} documents")
        skip_count = len(existing)
    else:
        existing   = set()
        skip_count = 0

    # Build list of (doc_id, data) to upload
    to_upload = []
    for _, row in df.iterrows():
        doc_id = row['date'].strftime('%Y-%m-%d')

        if doc_id in existing:
            continue  # skip if not overwriting

        # Build Firestore document
        doc_data = {
            'date': row['date'].to_pydatetime(),  # store as Firestore Timestamp
        }

        for csv_col, fb_field in FIELD_MAP.items():
            val = row[csv_col]
            if pd.isna(val):
                doc_data[fb_field] = None        # null in Firestore
            elif val == 0:
                doc_data[fb_field] = 0           # explicit zero (OOS)
            else:
                doc_data[fb_field] = round(float(val), 3)

        to_upload.append((doc_id, doc_data))

    print(f"\n  Records to upload: {len(to_upload)}")
    if skip_count:
        print(f"  Records skipped  : {skip_count} (already exist)")

    if not to_upload:
        print("\n  Nothing to upload. All records already exist in Firestore.")
        return 0

    # ── Batch upload ─────────────────────────────────────────────────────────
    uploaded    = 0
    failed_docs = []

    for batch_start in range(0, len(to_upload), BATCH_SIZE):
        batch_docs = to_upload[batch_start:batch_start + BATCH_SIZE]
        batch      = db.batch()

        for doc_id, doc_data in batch_docs:
            ref = db.collection(COLLECTION).document(doc_id)
            batch.set(ref, doc_data)

        try:
            batch.commit()
            uploaded += len(batch_docs)
            end_idx   = min(batch_start + BATCH_SIZE, len(to_upload))
            print(f"  ✅ Uploaded batch: records {batch_start + 1}–{end_idx} "
                  f"({uploaded}/{len(to_upload)})")
        except Exception as e:
            print(f"  ❌ Batch failed ({batch_start}–{batch_start+BATCH_SIZE}): {e}")
            failed_docs.extend([doc_id for doc_id, _ in batch_docs])

    return uploaded, failed_docs


# ─── VERIFY ──────────────────────────────────────────────────────────────────

def verify_upload(db, sample_dates: list):
    """Read back a few documents from Firestore to confirm the upload worked."""
    from firebase_admin import firestore

    print("\n  Verifying upload (reading back 3 documents)...")
    all_ok = True

    for date_str in sample_dates[:3]:
        doc = db.collection(COLLECTION).document(date_str).get()
        if doc.exists:
            data = doc.to_dict()
            petrol = data.get('92PetrolSale', 'MISSING')
            print(f"  ✅ {date_str}: 92PetrolSale={petrol}")
        else:
            print(f"  ❌ {date_str}: Document NOT found in Firestore!")
            all_ok = False

    return all_ok


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  BULK FIREBASE UPLOAD — EMERALD LANKA")
    print("=" * 60)
    print(f"  Target collection: {COLLECTION}")

    # ── Which file to upload? ─────────────────────────────────────────────
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV

    if not os.path.exists(csv_path):
        print(f"\n❌ File not found: {csv_path}")
        print(f"   Copy your CSV to: {DEFAULT_CSV}")
        sys.exit(1)

    # ── Load CSV ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading CSV...")
    df = load_csv(csv_path)

    # ── Preview ───────────────────────────────────────────────────────────
    preview(df)

    # ── Confirm ───────────────────────────────────────────────────────────
    print(f"\n  Ready to upload {len(df)} records to Firestore/{COLLECTION}")
    print(f"  Existing documents will be OVERWRITTEN with CSV values.")
    answer = input("\n  Proceed? (yes/no): ").strip().lower()

    if answer not in ('yes', 'y'):
        print("  Upload cancelled.")
        sys.exit(0)

    # ── Connect Firebase ──────────────────────────────────────────────────
    print("\n[2/4] Connecting to Firebase...")
    init_firebase()

    # ── Upload ────────────────────────────────────────────────────────────
    print(f"\n[3/4] Uploading to Firestore/{COLLECTION}...")
    started = datetime.now()
    uploaded, failed = upload_to_firebase(df, overwrite=True)
    elapsed = (datetime.now() - started).seconds

    # ── Verify ────────────────────────────────────────────────────────────
    print("\n[4/4] Verifying...")
    from firebase_admin import firestore
    db = firestore.client()
    sample = [df.iloc[0]['date'].strftime('%Y-%m-%d'),
              df.iloc[len(df)//2]['date'].strftime('%Y-%m-%d'),
              df.iloc[-1]['date'].strftime('%Y-%m-%d')]
    verify_upload(db, sample)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed:
        print(f"  ⚠️  Upload complete with errors")
        print(f"  Uploaded : {uploaded}/{len(df)}")
        print(f"  Failed   : {len(failed)} → {failed}")
    else:
        print(f"  ✅ UPLOAD COMPLETE in {elapsed}s")
        print(f"  Uploaded : {uploaded} records")
        print(f"  Collection: {COLLECTION}")

    print("\n  Next step — retrain models with the new data:")
    print("  python scripts/retrain.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()