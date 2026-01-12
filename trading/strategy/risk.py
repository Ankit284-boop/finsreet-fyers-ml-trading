def calc_qty(capital: float, price: float, position_pct: float) -> int:
    return int((capital * position_pct) / price)

def atr_sl_tp(entry_price: float, atr: float, sl_atr: float, tp_atr: float, side: int):
    """
    side: +1 for long, -1 for short
    """
    if side == 1:
        sl = entry_price - sl_atr * atr
        tp = entry_price + tp_atr * atr
    else:
        sl = entry_price + sl_atr * atr
        tp = entry_price - tp_atr * atr
    return sl, tp
