import pandas as pd

def prob_to_signal(p: float, buy_th: float, sell_th: float) -> int:
    if pd.isna(p):
        return 0
    if p >= buy_th:
        return 1
    if p <= sell_th:
        return -1
    return 0
