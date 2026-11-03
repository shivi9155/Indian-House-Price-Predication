"""Standalone script: python load_dataset.py"""
import pandas as pd
import db, model

CSV_FILE = "indian_urban_house_prices.csv"

def run_loader():
    print(f"Loading Kaggle dataset from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    print(f"Total CSV records available: {len(df)}")

    cleaned_1000 = model.clean_data(df).head(1000)
    print(f"Selected exactly 1,000 cleaned records: {len(cleaned_1000)}")

    status, msg = db.check_connection()
    if not status:
        print(f"MongoDB connection failed: {msg}")
        return

    count, info = db.init_house_data(cleaned_1000, force_reset=False)
    print(f"MongoDB status: {info}")
    print(f"Verified total documents in 'house_data': {db.get_total_records()}")

if __name__ == "__main__":
    run_loader()
