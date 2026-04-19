import os
import sys
import json

def setup_firebase_credentials():
    creds_json = os.environ.get('FIREBASE_CREDENTIALS')
    if not creds_json:
        print("⚠️  FIREBASE_CREDENTIALS not set")
        return False
    os.makedirs('firebase', exist_ok=True)
    try:
        cred_dict = json.loads(creds_json)
        with open('firebase/serviceAccountKey.json', 'w') as f:
            json.dump(cred_dict, f)
        print("✅ Firebase credentials written")
        return True
    except Exception as e:
        print(f"❌ Firebase credentials error: {e}")
        return False

def check_models():
    fuels = ['petrol_sales','super_petrol_sales','diesel_sales','super_diesel_sales']
    missing = [f for f in fuels if not os.path.exists(f'models/{f}_model.pkl')]
    return missing

def train_initial_models():
    print("🔄 Training models from CSV...")
    from scripts.preprocess import load_and_clean_data, save_clean_data
    from scripts.train_models import train_all_models
    os.makedirs('models', exist_ok=True)
    df = load_and_clean_data('data/sales_data_clean.csv')
    save_clean_data(df, 'data/sales_data_clean.csv')
    train_all_models()
    print("✅ Models trained!")

if __name__ == "__main__":
    print("🚀 Emerald Lanka — Starting up...")
    setup_firebase_credentials()
    missing = check_models()
    if missing:
        print(f"Missing models: {missing}")
        train_initial_models()
    else:
        print("✅ All 4 models found")
    print("✅ Startup complete!\n")