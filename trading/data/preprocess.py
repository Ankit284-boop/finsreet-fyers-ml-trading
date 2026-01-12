import pandas as pd

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["volume"] > 0].reset_index(drop=True)
    return df
