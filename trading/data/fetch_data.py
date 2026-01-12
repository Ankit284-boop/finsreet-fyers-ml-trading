import pandas as pd

def fetch_ohlcv_fyers(fyers_client, symbol: str, start_date: str, end_date: str, resolution="1D"):
    resp = fyers_client.history(
        symbol=symbol,
        resolution=resolution,
        range_from=start_date,
        range_to=end_date
    )

    if resp.get("s") != "ok":
        raise Exception(f"FYERS history fetch failed: {resp}")

    candles = resp["candles"]
    df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.drop(columns=["timestamp"]).sort_values("date").reset_index(drop=True)
    return df
