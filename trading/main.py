import yaml
import os
from dotenv import load_dotenv

from data.fyers_client import FyersClient
from data.fetch_data import fetch_ohlcv_fyers
from data.preprocess import clean_ohlcv
from features.indicators import add_features, add_target, get_feature_columns
from model.walk_forward import walk_forward_predict_proba
from strategy.signals import prob_to_signal
from backtest.backtest import run_backtest
from evaluation.metrics import sharpe_ratio, max_drawdown


def main():
    load_dotenv()

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    fyers = FyersClient(
        client_id=os.getenv("FYERS_CLIENT_ID"),
        access_token=os.getenv("FYERS_ACCESS_TOKEN")
    )

    df = fetch_ohlcv_fyers(
        fyers_client=fyers,
        symbol=cfg["symbol"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resolution="1D"
    )

    df = clean_ohlcv(df)
    df = add_features(df, atr_period=cfg["atr_period"])
    df = add_target(df)

    feature_cols = get_feature_columns()

    df = walk_forward_predict_proba(
        df=df,
        feature_cols=feature_cols,
        target_col="target",
        min_train=cfg["min_train"]
    )

    df["signal"] = df["prob_up"].apply(lambda p: prob_to_signal(p, cfg["buy_th"], cfg["sell_th"]))

    results = run_backtest(
        df=df,
        initial_capital=cfg["initial_capital"],
        position_pct=cfg["position_pct"],
        buy_th=cfg["buy_th"],
        sell_th=cfg["sell_th"],
        sl_atr=cfg["sl_atr"],
        tp_atr=cfg["tp_atr"]
    )

    equity = results["equity_curve"]
    rets = results["daily_returns"]

    print("\n===== RESULTS =====")
    print("Total Return:", round((equity.iloc[-1] / cfg["initial_capital"] - 1) * 100, 2), "%")
    print("Max Drawdown:", round(max_drawdown(equity) * 100, 2), "%")
    print("Sharpe Ratio:", round(sharpe_ratio(rets), 2))


if __name__ == "__main__":
    main()
