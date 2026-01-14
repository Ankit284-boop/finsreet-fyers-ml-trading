import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

TRAIN_START = "2025-11-03"
TRAIN_END = "2025-12-10"
TEST_END = "2025-12-31"

PROB_MIN = 0.05
PROB_MAX = 0.95

LONG_THRESHOLD = 0.72
SHORT_THRESHOLD = 0.28

DEPLOYMENT_DAYS = 5
DEFAULT_CSV_PATH = "data/RITES.csv"

st.set_page_config(page_title="RITES ML Trading Dashboard", page_icon="📈", layout="wide")

def inject_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(59,130,246,0.20), transparent 40%),
                    radial-gradient(circle at 90% 10%, rgba(168,85,247,0.20), transparent 35%),
                    radial-gradient(circle at 50% 90%, rgba(34,197,94,0.15), transparent 40%),
                    linear-gradient(120deg, #05080f 0%, #070a12 55%, #060816 100%);
        background-size: 200% 200%;
        animation: bgShift 14s ease-in-out infinite;
    }
    @keyframes bgShift {
        0% { background-position: 0% 30%; }
        50% { background-position: 100% 70%; }
        100% { background-position: 0% 30%; }
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { background: transparent !important; }
    .block-container { padding-top: 2.0rem !important; padding-bottom: 3rem !important; }
    h1 {
        font-weight: 900 !important;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleGlow 2.5s ease-in-out infinite alternate;
    }
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 0px rgba(96,165,250,0.0)); }
        to   { filter: drop-shadow(0 0 16px rgba(167,139,250,0.35)); }
    }
    .block-container > div { animation: fadeInUp 0.65s ease both; }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translate3d(0, 12px, 0); }
        to   { opacity: 1; transform: translate3d(0, 0, 0); }
    }
    section[data-testid="stSidebar"] > div {
        background: rgba(10, 12, 22, 0.55) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.12);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }
    .stButton > button {
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        padding: 0.7rem 1.1rem !important;
        font-weight: 800 !important;
        color: white !important;
        background: linear-gradient(90deg, #2563eb 0%, #9333ea 50%, #22c55e 100%) !important;
        background-size: 200% 200% !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        position: relative;
        overflow: hidden;
        animation: btnShift 3.5s ease infinite;
    }
    @keyframes btnShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stButton > button:hover { transform: translateY(-1px) scale(1.01); box-shadow: 0 18px 40px rgba(0,0,0,0.45); }
    .stButton > button:active { transform: translateY(1px) scale(0.995); }
    .stButton > button::after {
        content: "";
        position: absolute;
        top: -120%;
        left: -40%;
        width: 40%;
        height: 320%;
        transform: rotate(25deg);
        background: rgba(255,255,255,0.18);
        animation: shine 3.0s ease-in-out infinite;
    }
    @keyframes shine {
        0%   { left: -50%; opacity: 0; }
        35%  { opacity: 0.08; }
        55%  { left: 120%; opacity: 0; }
        100% { left: 120%; opacity: 0; }
    }
    div[data-testid="metric-container"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(148, 163, 184, 0.14) !important;
        padding: 18px !important;
        border-radius: 20px !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
        transition: transform 0.2s ease, border 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-4px); border: 1px solid rgba(96,165,250,0.35) !important; }
    div[data-testid="metric-container"]::before {
        content: "";
        position: absolute;
        inset: -2px;
        background: radial-gradient(circle at 20% 20%, rgba(96,165,250,0.14), transparent 60%),
                    radial-gradient(circle at 80% 30%, rgba(168,85,247,0.12), transparent 55%),
                    radial-gradient(circle at 50% 80%, rgba(34,197,94,0.10), transparent 60%);
        pointer-events: none;
    }
    div[data-testid="metric-container"] * { color: rgba(255,255,255,0.92) !important; position: relative; z-index: 2; }
    div[data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 16px 40px rgba(0,0,0,0.30);
        background: rgba(15, 23, 42, 0.55);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .js-plotly-plot {
        border-radius: 18px !important;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 20px 45px rgba(0,0,0,0.35);
    }
    .stMarkdown, .stCaption, p, li { color: rgba(226,232,240,0.88) !important; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

st.title("📈 RITES ML Trading Dashboard")
st.caption("Premium UI • Beginner-friendly • Output-focused")

st.sidebar.header("📤 Data Input")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("✅ Use default data (data/RITES.csv)", value=(uploaded is None))
run_btn = st.sidebar.button("🚀 Run Model & Generate Output", type="primary")

def feature_engineering(df):
    df = df.copy()
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    std_20 = df["close"].rolling(20).std()
    df["BB_upper"] = df["SMA_20"] + 2 * std_20
    df["BB_lower"] = df["SMA_20"] - 2 * std_20
    return df

def build_target(df):
    df = df.copy()
    df["return_1d"] = df["close"].pct_change().shift(-1)
    df["target"] = (df["return_1d"] > 0).astype(int)
    return df

def trade_signal(p):
    if p >= LONG_THRESHOLD:
        return 1
    if p <= SHORT_THRESHOLD:
        return -1
    return 0

def label_signal(sig):
    if sig == 1:
        return "✅ LONG"
    if sig == -1:
        return "🔻 SHORT"
    return "⏸ NO TRADE"

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Uploaded file selected ✅")
elif use_default:
    df = pd.read_csv(DEFAULT_CSV_PATH)
    st.sidebar.success(f"Using default file ✅ ({DEFAULT_CSV_PATH})")
else:
    st.warning("👈 Upload CSV or enable 'Use default data'.")
    st.stop()

required_cols = {"date", "open", "high", "low", "close", "volume"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV missing columns. Required: {required_cols}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df = df[df["volume"] > 0].reset_index(drop=True)

df = feature_engineering(df)
df = build_target(df)
df = df.dropna().reset_index(drop=True)

train_df = df.loc[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
test_df = df.loc[(df["date"] > TRAIN_END) & (df["date"] <= TEST_END)].copy()

features = ["SMA_20", "EMA_20", "RSI_14", "MACD", "MACD_signal", "BB_upper", "BB_lower"]

if train_df.empty:
    st.error("Training dataset is empty (check CSV date range).")
    st.stop()
if test_df.empty:
    st.error("Testing dataset is empty (check CSV date range).")
    st.stop()

X_train = train_df[features]
y_train = train_df["target"]
X_test = test_df[features]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

if not run_btn:
    st.info("Click **Run Model & Generate Output** in the sidebar.")
    st.stop()

with st.spinner("Running ML pipeline, generating signals and backtest..."):
    model.fit(X_train, y_train)
    test_df["prob_up"] = model.predict_proba(X_test)[:, 1]
    test_df["prob_up"] = test_df["prob_up"].clip(PROB_MIN, PROB_MAX)
    test_df["signal"] = test_df["prob_up"].apply(trade_signal)
    test_df["signal_label"] = test_df["signal"].apply(label_signal)
    test_df["strategy_return"] = test_df["signal"] * test_df["return_1d"]
    test_df["equity_curve"] = (1 + test_df["strategy_return"]).cumprod()

total_return = float(test_df["equity_curve"].iloc[-1] - 1)
drawdown = float((test_df["equity_curve"] / test_df["equity_curve"].cummax() - 1).min())

st.subheader("✅ Final Backtest Output")
c1, c2 = st.columns(2)
c1.metric("Total Return", f"{total_return*100:.2f}%")
c2.metric("Max Drawdown", f"{drawdown*100:.2f}%")

st.markdown("---")
left, right = st.columns(2)
with left:
    st.subheader("📊 Close Price (Test Period)")
    st.plotly_chart(px.line(test_df, x="date", y="close"), use_container_width=True)
with right:
    st.subheader("📈 Equity Curve")
    st.plotly_chart(px.line(test_df, x="date", y="equity_curve"), use_container_width=True)

st.markdown("---")

st.subheader("🚀 Jan 1–8 Signals")
deployment_df = df[df["date"] <= TEST_END].copy()
deployment_features = deployment_df[features].iloc[-DEPLOYMENT_DAYS:]
jan_probs = model.predict_proba(deployment_features)[:, 1]
jan_probs = np.clip(jan_probs, PROB_MIN, PROB_MAX)
jan_output = pd.DataFrame({
    "Day": range(1, DEPLOYMENT_DAYS + 1),
    "prob_up": np.round(jan_probs, 6),
    "signal": [label_signal(trade_signal(p)) for p in jan_probs]
})
st.dataframe(jan_output, use_container_width=True)

st.download_button(
    "⬇️ Download Deployment Signals CSV",
    data=jan_output.to_csv(index=False).encode("utf-8"),
    file_name="rites_deployment_signals.csv",
    mime="text/csv"
)


