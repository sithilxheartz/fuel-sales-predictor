import pandas as pd
import os

df = pd.read_csv('data/sales_data.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])
df = df.drop_duplicates(subset='date', keep='last')
df = df.sort_values('date').reset_index(drop=True)

df.to_csv('data/sales_data_clean.csv', index=False)
print(f'Reset done: {len(df)} rows')
print(f'Date range: {df["date"].min().date()} to {df["date"].max().date()}')