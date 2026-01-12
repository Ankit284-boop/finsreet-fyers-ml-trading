import streamlit as st
import plotly.graph_objects as go
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


st.set_page_config(page_title="FYERS ML Trading Dashboard", layout="wide")
st.title("📊 FYERS ML Trading System Dashboard")

load_dotenv()

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

st.sidebar.header("Controls")
buy_th = st.sidebar.slider("BUY threshold", 0.55, 0.80, float(cfg["buy_th"]), 0.01)
sell_th = st.sidebar.slider("SELL threshold", 0.20, 0.45, float(cfg["sell_th"]), 0.01)
position_pct = st.sidebar.slider("Position %", 0.05, 0.30, float(cfg["position_pct"]), 0.01)

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
df = walk_forward_predict_proba(df, feature_cols, "target", cfg["min_train"])

df["signal"] = df["prob_up"].apply(lambda p: prob_to_signal(p, buy_th, sell_th))

results = run_backtest(
    df=df,
    initial_capital=cfg["initial_capital"],
    position_pct=position_pct,
    buy_th=buy_th,
    sell_th=sell_th,
    sl_atr=cfg["sl_atr"],
    tp_atr=cfg["tp_atr"]
)

equity = results["equity_curve"]
rets = results["daily_returns"]
trades = results["trades"]

c1, c2, c3 = st.columns(3)
c1.metric("Sharpe Ratio", f"{sharpe_ratio(rets):.2f}")
c2.metric("Max Drawdown", f"{max_drawdown(equity)*100:.2f}%")
c3.metric("Total Return", f"{(equity.iloc[-1]/cfg['initial_capital']-1)*100:.2f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=equity.index, y=equity.values, mode="lines", name="Equity Curve"))
fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Latest Signals")
view_df = df[["date", "close", "prob_up", "signal"]].tail(20).copy()
view_df["signal_label"] = view_df["signal"].map({1: "LONG", -1: "SHORT", 0: "HOLD"})
st.dataframe(view_df)

st.subheader("Trades")
st.dataframe(trades.tail(30))
