import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV from Kaggle and do basic parsing."""
    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()
    print(f"✅ Loaded {len(df):,} rows from {os.path.basename(filepath)}")
    print(f"   Date range: {df.index.min()} → {df.index.max()}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and outliers."""
    # Drop duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Fill missing with interpolation
    df = df.interpolate(method="time")

    # Remove extreme outliers (beyond 3 std devs)
    col = df.columns[0]  # energy column
    mean, std = df[col].mean(), df[col].std()
    df = df[(df[col] >= mean - 3 * std) & (df[col] <= mean + 3 * std)]

    print(f"✅ Cleaned data: {len(df):,} rows remaining")
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features useful for forecasting."""
    df = df.copy()
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek      # 0=Monday
    df["month"]       = df.index.month
    df["year"]        = df.index.year
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"]     = df.index.quarter

    # Cyclical encoding for hour and month (better for ML models)
    import numpy as np
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print("✅ Time features added")
    return df
