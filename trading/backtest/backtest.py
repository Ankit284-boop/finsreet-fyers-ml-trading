import pandas as pd

def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    position_pct: float,
    buy_th: float,
    sell_th: float,
    sl_atr: float,
    tp_atr: float,
):
    capital = initial_capital
    position = 0
    entry_price = None
    sl = None
    tp = None

    equity_curve = []
    daily_returns = []
    trades = []
    prev_equity = initial_capital

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        sig = prev_row["signal"]
        prob = prev_row["prob_up"]

        open_p = row["open"]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        atr = prev_row["ATR"]

        # EXIT
        if position != 0:
            exit_price, reason = None, None

            if position > 0 and low <= sl:
                exit_price, reason = sl, "SL"
            elif position < 0 and high >= sl:
                exit_price, reason = sl, "SL"
            elif position > 0 and high >= tp:
                exit_price, reason = tp, "TP"
            elif position < 0 and low <= tp:
                exit_price, reason = tp, "TP"
            elif (position > 0 and prob <= sell_th) or (position < 0 and prob >= buy_th):
                exit_price, reason = close, "FLIP"

            if exit_price is not None:
                pnl = position * (exit_price - entry_price)
                capital += pnl
                trades.append({
                    "date": row["date"],
                    "action": "EXIT",
                    "price": exit_price,
                    "qty": position,
                    "pnl": pnl,
                    "reason": reason
                })
                position, entry_price, sl, tp = 0, None, None, None

        # ENTRY
        if position == 0 and pd.notna(prob):
            qty = int((capital * position_pct) / open_p)

            if sig == 1 and qty > 0:
                position = qty
                entry_price = open_p
                sl = entry_price - sl_atr * atr
                tp = entry_price + tp_atr * atr
                trades.append({"date": row["date"], "action": "BUY", "price": entry_price, "qty": qty, "prob_up": prob})

            elif sig == -1 and qty > 0:
                position = -qty
                entry_price = open_p
                sl = entry_price + sl_atr * atr
                tp = entry_price - tp_atr * atr
                trades.append({"date": row["date"], "action": "SELL", "price": entry_price, "qty": -qty, "prob_up": prob})

        equity = capital + position * close
        equity_curve.append(equity)

        ret = (equity - prev_equity) / prev_equity
        daily_returns.append(ret)
        prev_equity = equity

    return {
        "equity_curve": pd.Series(equity_curve, index=df["date"].iloc[1:]),
        "daily_returns": pd.Series(daily_returns, index=df["date"].iloc[1:]),
        "trades": pd.DataFrame(trades)
    }
