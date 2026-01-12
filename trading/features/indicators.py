import pandas as pd

def add_features(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()

    df["SMA_20"] = df["close"].rolling(20).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    std_20 = df["close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * std_20
    df["BB_lower"] = df["SMA_20"] - 2 * std_20

    # ATR
    df["HL"] = df["high"] - df["low"]
    df["ATR"] = df["HL"].rolling(atr_period).mean()

    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1d_next"] = df["close"].pct_change().shift(-1)
    df["target"] = (df["return_1d_next"] > 0).astype(int)
    return df.dropna().reset_index(drop=True)

def get_feature_columns():
    return [
        "SMA_20", "EMA_20", "RSI_14",
        "MACD", "MACD_signal",
        "BB_upper", "BB_lower",
        "ATR"
    ]
