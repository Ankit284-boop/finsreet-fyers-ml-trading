import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ======================
# Load & clean data
# ======================
df = pd.read_csv("data/RITES.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Remove zero-volume days (VERY IMPORTANT)
df = df[df['volume'] > 0].reset_index(drop=True)

# ======================
# Feature Engineering
# ======================
df['SMA_20'] = df['close'].rolling(20).mean()
df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

# RSI
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df['close'].ewm(span=12, adjust=False).mean()
ema_26 = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
std_20 = df['close'].rolling(20).std()
df['BB_upper'] = df['SMA_20'] + 2 * std_20
df['BB_lower'] = df['SMA_20'] - 2 * std_20

# ======================
# Target Variable
# ======================
df['return_1d'] = df['close'].pct_change().shift(-1)
df['target'] = (df['return_1d'] > 0).astype(int)

df = df.dropna().reset_index(drop=True)

# ======================
# Train / Test Split
# ======================
train_df = df.loc[
    (df['date'] >= "2025-11-03") & (df['date'] <= "2025-12-10")
].copy()

test_df = df.loc[
    (df['date'] > "2025-12-10") & (df['date'] <= "2025-12-31")
].copy()

features = [
    'SMA_20', 'EMA_20', 'RSI_14',
    'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower'
]

X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]

# ======================
# Model
# ======================
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

model.fit(X_train, y_train)

# ======================
# Predictions
# ======================
test_df['prob_up'] = model.predict_proba(X_test)[:, 1]
test_df['prob_up'] = test_df['prob_up'].clip(0.05, 0.95)
test_df['prob_down'] = 1 - test_df['prob_up']

# ======================
# Trading Logic
# ======================
def signal(p):
    if p >= 0.65:
        return 1     # LONG
    elif p <= 0.35:
        return -1    # SHORT
    else:
        return 0     # NO TRADE

test_df['signal'] = test_df['prob_up'].apply(signal)

# ======================
# Backtest
# ======================
test_df['strategy_return'] = test_df['signal'] * test_df['return_1d']

test_df['equity_curve'] = (1 + test_df['strategy_return']).cumprod()

# Metrics
total_return = test_df['equity_curve'].iloc[-1] - 1
drawdown = (
    test_df['equity_curve'] /
    test_df['equity_curve'].cummax() - 1
).min()

sharpe = (
    test_df['strategy_return'].mean() /
    test_df['strategy_return'].std()
) * np.sqrt(252)

print("Total Return:", round(total_return * 100, 2), "%")
print("Max Drawdown:", round(drawdown * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe, 2))

print(test_df[['date', 'close', 'prob_up', 'signal']])




# ======================
# JAN 1–8 SIGNAL GENERATION (DEPLOYMENT)
# ======================

# IMPORTANT:
# Model is NOT retrained here.
# We reuse the trained model and last available data (till Dec 31).

deployment_df = df[df['date'] <= "2025-12-31"].copy()

# Use last 5 trading days to represent Jan 1–8 predictions
deployment_features = deployment_df[features].iloc[-5:]

jan_probs = model.predict_proba(deployment_features)[:, 1]
jan_probs = np.clip(jan_probs, 0.05, 0.95)

def deploy_signal(p):
    if p >= 0.65:
        return "LONG"
    elif p <= 0.35:
        return "SHORT"
    else:
        return "NO TRADE"

jan_signals = [deploy_signal(p) for p in jan_probs]

jan_output = pd.DataFrame({
    "Day": range(1, len(jan_signals) + 1),
    "prob_up": jan_probs,
    "signal": jan_signals
})

print("\nJAN 1–8 TRADE SIGNALS (NO LOOKAHEAD)")
print(jan_output)
