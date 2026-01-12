import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1
    return float(dd.min())

def sharpe_ratio(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0:
        return 0.0
    return float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
