"""
FASTAPI BACKEND — EMERALD LANKA FUEL PREDICTION
================================================
Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
  GET  /                    → welcome message
  GET  /health              → server + model status
  GET  /predict/tomorrow    → next day prediction (all 4 fuels)
  GET  /predict/7days       → 7-day forecast (all 4 fuels)
  GET  /predict/summary     → combined tomorrow + 7days in one call
  POST /retrain             → trigger background retraining from Firebase
"""

import os
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from scripts.predict import load_models, get_tomorrow_prediction, get_7day_prediction
from scripts.preprocess import load_and_clean_data, save_clean_data


# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emerald Lanka Fuel Prediction API",
    description="Fuel sales forecasting for Emerald Lanka Filling Station, Hettipola",
    version="1.0.0"
)

# Allow all origins so your Flutter app can connect from any device/network
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── LOAD MODELS + DATA AT STARTUP ───────────────────────────────────────────

print("\n🚀 Starting Emerald Lanka Prediction API...")
print("   Loading models...")
MODELS = load_models("models")

print("   Loading historical data...")
HISTORICAL_DATA = pd.read_csv("data/sales_data_clean.csv")
HISTORICAL_DATA['date'] = pd.to_datetime(HISTORICAL_DATA['date'])

RETRAIN_STATUS = {"running": False, "last_run": None, "last_result": None}

print(f"   ✅ API ready! {len(MODELS)} models loaded, {len(HISTORICAL_DATA)} data rows\n")


# ─── RESPONSE SCHEMAS (Pydantic models) ──────────────────────────────────────

class FuelPrediction(BaseModel):
    date: str
    predicted_litres: float

class TomorrowFuel(BaseModel):
    petrol: FuelPrediction
    super_petrol: FuelPrediction
    diesel: FuelPrediction
    super_diesel: FuelPrediction
    total_litres: float

class TomorrowResponse(BaseModel):
    generated_at: str
    data: TomorrowFuel
    accuracy_notes: dict

class SevenDayResponse(BaseModel):
    generated_at: str
    petrol: List[FuelPrediction]
    super_petrol: List[FuelPrediction]
    diesel: List[FuelPrediction]
    super_diesel: List[FuelPrediction]

class SummaryResponse(BaseModel):
    generated_at: str
    tomorrow: TomorrowFuel
    seven_days: dict
    accuracy_notes: dict


# ─── HELPER ──────────────────────────────────────────────────────────────────

def build_tomorrow_fuel(preds: dict) -> TomorrowFuel:
    p  = preds['petrol_sales']['predicted_litres']
    sp = preds['super_petrol_sales']['predicted_litres']
    d  = preds['diesel_sales']['predicted_litres']
    sd = preds['super_diesel_sales']['predicted_litres']
    date = preds['petrol_sales']['date']

    return TomorrowFuel(
        petrol=FuelPrediction(date=date, predicted_litres=p),
        super_petrol=FuelPrediction(date=date, predicted_litres=sp),
        diesel=FuelPrediction(date=date, predicted_litres=d),
        super_diesel=FuelPrediction(date=date, predicted_litres=sd),
        total_litres=round(p + sp + d + sd, 2)
    )

ACCURACY_NOTES = {
    "petrol":       {"mape_pct": 7.21,  "reliability": "high",   "note": "Reliable for ordering decisions"},
    "super_petrol": {"mape_pct": 49.62, "reliability": "low",    "note": "Rough estimate — high OOS days in training data"},
    "diesel":       {"mape_pct": 13.62, "reliability": "medium", "note": "Good — acceptable for planning"},
    "super_diesel": {"mape_pct": 62.88, "reliability": "low",    "note": "Rough estimate — erratic training data"},
}


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "station": "Emerald Lanka Filling Station, Hettipola",
        "api": "Fuel Sales Prediction API v1.0",
        "docs": "/docs",
        "endpoints": ["/health", "/predict/tomorrow", "/predict/7days", "/predict/summary", "/retrain"]
    }


@app.get("/health")
def health_check():
    """Check server status, loaded models and data freshness."""
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "model_count": len(MODELS),
        "data_rows": len(HISTORICAL_DATA),
        "data_latest_date": str(HISTORICAL_DATA['date'].max().date()),
        "retrain_running": RETRAIN_STATUS["running"],
        "last_retrain": RETRAIN_STATUS["last_run"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/predict/tomorrow", response_model=TomorrowResponse)
def predict_tomorrow():
    """
    Returns tomorrow's fuel sales prediction for all 4 fuel types.

    Example response:
    {
      "generated_at": "2026-02-01T09:00:00",
      "data": {
        "petrol":       {"date": "2026-02-02", "predicted_litres": 2508.6},
        "super_petrol": {"date": "2026-02-02", "predicted_litres": 106.2},
        "diesel":       {"date": "2026-02-02", "predicted_litres": 2868.6},
        "super_diesel": {"date": "2026-02-02", "predicted_litres": 144.5},
        "total_litres": 5628.0
      },
      "accuracy_notes": { ... }
    }
    """
    try:
        preds = get_tomorrow_prediction(HISTORICAL_DATA, MODELS)
        return TomorrowResponse(
            generated_at=datetime.now().isoformat(),
            data=build_tomorrow_fuel(preds),
            accuracy_notes=ACCURACY_NOTES
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/7days", response_model=SevenDayResponse)
def predict_7_days():
    """
    Returns 7-day fuel sales forecast for all 4 fuel types.
    """
    try:
        preds = get_7day_prediction(HISTORICAL_DATA, MODELS)

        def to_list(fuel_key):
            return [FuelPrediction(**p) for p in preds[fuel_key]]

        return SevenDayResponse(
            generated_at=datetime.now().isoformat(),
            petrol=to_list('petrol_sales'),
            super_petrol=to_list('super_petrol_sales'),
            diesel=to_list('diesel_sales'),
            super_diesel=to_list('super_diesel_sales'),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/summary")
def predict_summary():
    """
    Combined endpoint — returns BOTH tomorrow and 7-day forecast in one call.
    Use this in Flutter to load the entire dashboard with a single HTTP request.
    Faster and more efficient than calling two endpoints separately.
    """
    try:
        tomorrow_preds = get_tomorrow_prediction(HISTORICAL_DATA, MODELS)
        seven_day_preds = get_7day_prediction(HISTORICAL_DATA, MODELS)

        return {
            "generated_at": datetime.now().isoformat(),
            "data_as_of": str(HISTORICAL_DATA['date'].max().date()),
            "tomorrow": {
                "petrol":       tomorrow_preds['petrol_sales'],
                "super_petrol": tomorrow_preds['super_petrol_sales'],
                "diesel":       tomorrow_preds['diesel_sales'],
                "super_diesel": tomorrow_preds['super_diesel_sales'],
                "total_litres": round(sum(
                    v['predicted_litres'] for v in tomorrow_preds.values()
                ), 2)
            },
            "seven_days": {
                "petrol":       seven_day_preds['petrol_sales'],
                "super_petrol": seven_day_preds['super_petrol_sales'],
                "diesel":       seven_day_preds['diesel_sales'],
                "super_diesel": seven_day_preds['super_diesel_sales'],
            },
            "accuracy_notes": ACCURACY_NOTES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Triggers model retraining in the background.
    Fetches new data from Firebase → preprocesses → retrains all 4 models.
    Check /health after a few minutes to confirm completion.
    """
    if RETRAIN_STATUS["running"]:
        return {
            "message": "Retraining already in progress. Check /health for status.",
            "started_at": RETRAIN_STATUS["last_run"]
        }

    RETRAIN_STATUS["running"] = True
    RETRAIN_STATUS["last_run"] = datetime.now().isoformat()

    background_tasks.add_task(_run_retrain_pipeline)

    return {
        "message": "✅ Retraining started. Check /health in 2–3 minutes.",
        "started_at": RETRAIN_STATUS["last_run"]
    }


def _run_retrain_pipeline():
    """Background task: fetch Firebase data → preprocess → retrain → reload."""
    global HISTORICAL_DATA, MODELS
    try:
        print("\n🔄 Retraining pipeline started...")

        # Step 1: Fetch new data from Firebase and merge with existing
        try:
            from scripts.retrain import fetch_from_firestore, merge_with_historical, save_merged
            firebase_df = fetch_from_firestore()
            merged_df   = merge_with_historical(firebase_df)
            save_merged(merged_df, 'data/sales_data_merged.csv')
            data_path = 'data/sales_data_merged.csv'
            print("✅ Firebase data fetched and merged")
        except Exception as e:
            print(f"   ⚠️  Firebase fetch failed ({e}), retraining on existing data")
            data_path = 'data/sales_data_clean.csv'

        # Step 2: Preprocess
        from scripts.preprocess import load_and_clean_data, save_clean_data
        df = load_and_clean_data(data_path)
        save_clean_data(df, 'data/sales_data_clean.csv')

        # Step 3: Retrain all 4 models
        from scripts.train_models import train_all_models
        train_all_models()

        # Step 4: Reload into memory
        MODELS = load_models('models')
        HISTORICAL_DATA = pd.read_csv('data/sales_data_clean.csv')
        HISTORICAL_DATA['date'] = pd.to_datetime(HISTORICAL_DATA['date'])

        RETRAIN_STATUS['running'] = False
        RETRAIN_STATUS['last_result'] = 'success'
        print("✅ Retraining complete!\n")

    except Exception as e:
        RETRAIN_STATUS['running'] = False
        RETRAIN_STATUS['last_result'] = f'failed: {str(e)}'
        print(f"❌ Retraining failed: {e}\n")