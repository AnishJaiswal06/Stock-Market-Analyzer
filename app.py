#!/usr/bin/env python3
"""
Stock Market Analyzer — Premium Streamlit Dashboard
=====================================================
Launch:  streamlit run app.py
"""

import sys
import os

# Ensure project root is on sys.path so absolute imports work from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from dotenv import load_dotenv
import torch

# --- STREAMLIT + PYTORCH BUG FIX ---
try:
    torch._classes.__path__ = []
except Exception:
    pass
# -----------------------------------
from torch.utils.data import DataLoader
import logging
from sklearn.preprocessing import MinMaxScaler

from data_pipeline import fetch_ohlcv, fetch_fundamentals, align_and_merge
from features import calculate_technical_indicators, create_target_labels, scale_features
from models import StockDataset, LSTMModel, train_model, evaluate_model, train_random_forest
from backtest import run_backtest, calculate_performance_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20", "EMA_50",
    "RSI_14", "MACD", "Signal_Line", "MACD_Hist",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "Vol_20", "Momentum_10", "Daily_Returns", "Sector_Return",
]

# ---------------------------------------------------------------------------
# Color palette — clean financial light theme
# ---------------------------------------------------------------------------
C_NAVY      = "#0f172a"          # sidebar / dark accents (slate-900)
C_BLUE      = "#4f46e5"          # primary accent (indigo-600)
C_EMERALD   = "#10b981"          # positive / emerald-500
C_AMBER     = "#d97706"          # warning / amber-600
C_PURPLE    = "#7c3aed"          # secondary accent (violet-600)
C_RED       = "#ef4444"          # negative / red-500
C_SURFACE   = "#ffffff"          # card surface
C_BORDER    = "#e2e8f0"          # subtle border (slate-200)
C_TEXT      = "#0f172a"          # primary text (slate-900)
C_TEXT_DIM  = "#64748b"          # muted text (slate-500)
C_BG        = "#f8fafc"          # main content bg (slate-50)
C_ORANGE    = "#ea580c"          # chart accent orange-600

# ---------------------------------------------------------------------------
# Inject custom CSS — clean, modern financial dashboard
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ---------- root overrides ---------- */
:root {{
    --bg-main: {C_BG};
    --bg-card: {C_SURFACE};
    --accent: {C_BLUE};
    --emerald: {C_EMERALD};
    --amber: {C_AMBER};
    --purple: {C_PURPLE};
    --red: {C_RED};
    --text-primary: {C_TEXT};
    --text-dim: {C_TEXT_DIM};
    --border: {C_BORDER};
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: {C_BG} !important;
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* ---- sidebar ---- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C_NAVY} 0%, #0f172a 100%) !important;
    border-right: none;
}}
[data-testid="stSidebar"] * {{
    color: #cbd5e1 !important;
}}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: #f1f5f9 !important;
}}
[data-testid="stSidebar"] .stMarkdown hr {{
    border-color: rgba(255,255,255,0.08) !important;
}}
/* sidebar inputs */
[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stDateInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #f1f5f9 !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px;
}}

/* ---- headings ---- */
h1, h2, h3, h4 {{
    color: {C_TEXT} !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
}}

/* ---- metric cards ---- */
[data-testid="stMetric"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 4px 20px -2px rgba(79, 70, 229, 0.1);
    transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-4px);
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.15), 0 8px 10px -6px rgba(79, 70, 229, 0.1);
}}
[data-testid="stMetric"] label {{
    color: {C_TEXT_DIM} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.68rem !important;
    letter-spacing: 0.07em;
}}
[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {C_TEXT} !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
}}
/* positive/negative delta coloring */
[data-testid="stMetric"] [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] {{
    fill: {C_EMERALD} !important;
}}
[data-testid="stMetric"] [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] {{
    fill: {C_RED} !important;
}}
[data-testid="stMetric"] [data-testid="stMetricDelta"][style*="color"] {{
    font-weight: 600 !important;
}}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: transparent;
    border-bottom: none;
    padding-bottom: 12px;
}}
.stTabs [data-baseweb="tab"] {{
    background: {C_BORDER};
    border: none;
    border-radius: 8px;
    color: {C_TEXT};
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s ease-out;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: #cbd5e1;
    color: {C_NAVY};
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(to right, {C_BLUE}, {C_PURPLE}) !important;
    border-bottom: none !important;
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.3);
    transform: translateY(-1px);
}}

/* ---- buttons ---- */
.stButton > button {{
    background: linear-gradient(to right, {C_BLUE}, {C_PURPLE});
    color: #ffffff;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.3);
    transition: all 0.2s ease-out;
    letter-spacing: 0.02em;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.4);
}}
.stButton > button:active {{
    transform: translateY(0);
    box-shadow: 0 2px 8px 0 rgba(79, 70, 229, 0.3);
}}

/* ---- expander (main area) ---- */
[data-testid="stExpander"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 12px;
}}

/* ---- dataframe / table — minimalist ---- */
.stDataFrame {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {C_BORDER};
}}
.stDataFrame table {{
    border-collapse: collapse;
}}
.stDataFrame th {{
    background: #f1f5f9 !important;
    color: {C_TEXT_DIM} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em;
    border-bottom: 2px solid {C_BORDER} !important;
    border-left: none !important;
    border-right: none !important;
    padding: 14px 16px !important;
}}
.stDataFrame td {{
    border-bottom: 1px solid #f1f5f9 !important;
    border-left: none !important;
    border-right: none !important;
    padding: 12px 16px !important;
    color: {C_TEXT} !important;
    font-size: 0.85rem;
}}
.stDataFrame tr:hover td {{
    background: #f8fafc !important;
}}

/* ---- slider ---- */
.stSlider > div > div > div {{
    color: {C_BLUE};
}}

/* ---- text inputs (main area) ---- */
.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stNumberInput > div > div > input {{
    background: {C_SURFACE} !important;
    border: 1px solid {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 8px !important;
}}

/* ---- scrollbar ---- */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: #cbd5e1;
    border-radius: 4px;
}}

/* ---- hero section ---- */
.hero-title {{
    font-size: 2.1rem;
    font-weight: 800;
    color: {C_TEXT} !important;
    -webkit-text-fill-color: {C_TEXT};
    margin-bottom: 0;
    line-height: 1.15;
}}
.hero-sub {{
    color: {C_TEXT_DIM};
    font-size: 0.95rem;
    margin-top: 2px;
}}

/* ---- info tooltip icon ---- */
.info-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}}
.info-header h4 {{
    margin: 0 !important;
    padding: 0 !important;
}}
.info-icon-wrapper {{
    position: relative;
    display: inline-flex;
    align-items: center;
    cursor: help;
}}
.info-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: rgba(79, 70, 229, 0.08);
    border: 1px solid rgba(79, 70, 229, 0.2);
    color: {C_BLUE};
    font-size: 0.72rem;
    font-weight: 700;
    font-style: normal;
    transition: all 0.25s ease;
    flex-shrink: 0;
}}
.info-icon-wrapper:hover .info-icon {{
    background: rgba(37, 99, 235, 0.15);
    border-color: {C_BLUE};
    box-shadow: 0 0 8px rgba(37, 99, 235, 0.15);
    transform: scale(1.1);
}}
.info-tooltip {{
    visibility: hidden;
    opacity: 0;
    position: absolute;
    left: 32px;
    top: 50%;
    transform: translateY(-50%);
    min-width: 280px;
    max-width: 360px;
    padding: 14px 18px;
    background: {C_SURFACE};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    color: {C_TEXT};
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.55;
    letter-spacing: 0.01em;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    z-index: 9999;
    transition: opacity 0.22s ease, visibility 0.22s ease;
    pointer-events: none;
}}
.info-icon-wrapper:hover .info-tooltip {{
    visibility: visible;
    opacity: 1;
}}
.info-tooltip strong {{
    color: {C_BLUE};
}}

/* ---- pill badges for signal column ---- */
.pill-buy {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(22,163,74,0.1);
    color: {C_EMERALD};
}}
.pill-hold {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(217,119,6,0.1);
    color: {C_AMBER};
}}
.pill-sell {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(220,38,38,0.1);
    color: {C_RED};
}}

/* ---- radial gauge ---- */
.radial-gauge {{
    position: relative;
    width: 140px;
    height: 140px;
    margin: 0 auto;
}}
.radial-gauge svg {{
    transform: rotate(-90deg);
}}
.radial-gauge .gauge-label {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    text-align: center;
}}

/* ---- sidebar rectangular tabs ---- */
[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap: 0.5rem;
}}
[data-testid="stSidebar"] div[role="radiogroup"] > label {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
}}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
    background: rgba(255, 255, 255, 0.1);
}}
[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] div[role="radiogroup"] > label[aria-checked="true"] {{
    background: #2563eb !important;
    border-color: #2563eb !important;
}}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {{
    display: none; /* hide the native radio circle */
}}
[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
    color: #f1f5f9 !important;
    font-weight: 600;
    margin: 0;
    font-size: 0.95rem;
}}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly layout defaults — light financial theme
# ---------------------------------------------------------------------------
_PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=48, r=20, t=40, b=40),
    xaxis=dict(
        showgrid=False,
        linecolor=C_BORDER,
        linewidth=1,
        tickfont=dict(color=C_TEXT, size=11),
        title_font=dict(color=C_TEXT, size=13),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(226,232,240,0.6)",
        gridwidth=1,
        griddash="solid",
        linecolor=C_BORDER,
        linewidth=1,
        tickfont=dict(color=C_TEXT, size=11),
        title_font=dict(color=C_TEXT, size=13),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=C_BORDER,
        borderwidth=1,
        font=dict(size=11, color=C_TEXT),
    ),
    title_font=dict(color=C_TEXT, size=16),
)


def _apply_layout(fig: go.Figure, **kw) -> go.Figure:
    """Apply common light financial layout to a plotly figure."""
    merged = {**_PLOTLY_LAYOUT, **kw}
    fig.update_layout(**merged)
    # Enforce dark x-axis text on all subplots explicitly
    fig.update_xaxes(
        tickfont=dict(color=C_TEXT, size=11),
        title_font=dict(color=C_TEXT, size=13)
    )
    return fig


def _info_header(title: str, tooltip: str, level: str = "####") -> None:
    """Render a section heading with a hoverable ⓘ info tooltip icon."""
    st.markdown(
        f'<div class="info-header">'
        f'<h4>{title}</h4>'
        f'<span class="info-icon-wrapper">'
        f'<span class="info-icon">i</span>'
        f'<span class="info-tooltip">{tooltip}</span>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cached data fetchers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_ohlcv(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    return fetch_ohlcv(ticker, start_date=start, end_date=end)


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_fundamentals(ticker: str, email: str) -> pd.DataFrame | None:
    return fetch_fundamentals(ticker, user_agent_email=email)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar() -> dict:
    """Render sidebar controls and return parameter dict."""
    with st.sidebar:
        st.markdown("## 📈 Stock Analyzer")
        app_mode = st.radio("Navigation", ["Analyzer", "Screener", "Admin Data"], horizontal=False, label_visibility="collapsed")
        st.markdown("---")

        if app_mode == "Screener":
            st.markdown(
                "<div style='color:var(--text-dim); font-size:0.9rem;'>"
                "Scan the S&P 500 for the best technical and AI-driven trading setups. "
                "Select criteria in the main view to run the screener."
                "</div>",
                unsafe_allow_html=True
            )
            return {"mode": "Screener", "run": False}
        elif app_mode == "Admin Data":
            st.markdown(
                "<div style='color:var(--text-dim); font-size:0.9rem;'>"
                "View system logs and pipeline execution status."
                "</div>",
                unsafe_allow_html=True
            )
            return {"mode": "Admin Data", "run": False}

        ticker = st.text_input("Ticker Symbol", value="AAPL", help="e.g. AAPL, MSFT, TSLA, SPY")

        st.markdown("##### 📅 Date Range")
        col1, col2 = st.columns(2)
        default_end = date.today() - timedelta(days=1)
        default_start = default_end - timedelta(days=900)
        with col1:
            start_date = st.date_input("Start", value=default_start)
        with col2:
            end_date = st.date_input("End", value=default_end)

        st.markdown("---")

        with st.expander("🧠 Model Parameters", expanded=False):
            seq_len = st.slider("Sequence Length", 10, 120, 60, step=5)
            epochs = st.slider("Epochs", 1, 100, 15)
            batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128, 256], value=64)
            hidden_size = st.select_slider("Hidden Size", options=[32, 64, 128, 256], value=64)
            num_layers = st.slider("LSTM Layers", 1, 4, 2)
            dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001,
                format_func=lambda x: f"{x:.4f}",
            )
            horizon = st.slider("Prediction Horizon (days)", 1, 30, 5)
            cls_threshold = st.slider("Classification Threshold", 0.0, 0.05, 0.01, step=0.005)

        with st.expander("💰 Backtest Parameters", expanded=False):
            initial_capital = st.number_input("Initial Capital ($)", value=10000.0, step=1000.0, min_value=100.0)
            signal_threshold = st.slider("Signal Threshold", 0.0, 1.0, 0.5, step=0.05)
            risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, 0.02, step=0.005, format="%.3f")

        sec_email = st.text_input("SEC User-Agent Email", value="", help="Required for fundamental data")

        st.markdown("---")
        run_clicked = st.button("🚀  Run Analysis", use_container_width=True)

    return {
        "mode": "Analyzer",
        "ticker": ticker.upper().strip(),
        "start_date": start_date,
        "end_date": end_date,
        "sequence_length": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "prediction_horizon": horizon,
        "classification_threshold": cls_threshold,
        "initial_capital": initial_capital,
        "signal_threshold": signal_threshold,
        "risk_free_rate": risk_free_rate,
        "sec_email": sec_email.strip(),
        "run": run_clicked,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═══════════════════════════════════════════════════════════════════════════
def render_overview(ohlcv_df: pd.DataFrame, fund_df: pd.DataFrame | None, ticker: str):
    """Tab 1 — price overview, candlestick chart, fundamentals."""

    # --- Hero metrics ---
    latest = ohlcv_df.iloc[-1]
    prev = ohlcv_df.iloc[-2] if len(ohlcv_df) > 1 else latest
    
    # Format the exact date of the latest price
    latest_date_str = latest.name.strftime("%b %d, %Y")
    st.markdown(
        f"### {ticker} — Market Overview "
        f"<span style='font-size:0.95rem; color:{C_TEXT_DIM}; font-weight:normal; margin-left:12px;'>"
        f"(As of {latest_date_str})</span>", 
        unsafe_allow_html=True
    )
    change_pct = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
    high52 = ohlcv_df["High"].rolling(252, min_periods=1).max().iloc[-1]
    low52 = ohlcv_df["Low"].rolling(252, min_periods=1).min().iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"${latest['Close']:.2f}", f"{change_pct:+.2f}%",
             help="The most recent closing price of the stock. The delta shows the percentage change from the previous trading day.")
    m2.metric("Daily Change", f"{change_pct:+.2f}%", f"{'▲' if change_pct >= 0 else '▼'}",
             help="Percentage change in closing price compared to the prior trading day. Green ▲ means the price went up; red ▼ means it went down.")
    m3.metric("52-Week Range", f"${low52:.2f} – ${high52:.2f}",
             help="The lowest and highest prices the stock reached over the past 252 trading days (~1 year). Useful for gauging where the current price sits relative to its recent history.")
    m4.metric("Volume", f"{latest['Volume']:,.0f}",
             help="Total number of shares traded on the most recent trading day. High volume often confirms the strength of a price move.")

    st.markdown("")

    # --- Candlestick + Volume ---
    _info_header("📊 Price & Volume",
                 "<strong>Candlestick chart</strong> showing daily Open/High/Low/Close prices. "
                 "Green candles = price went up; red = went down. The <strong>volume bars</strong> below show "
                 "trading activity — taller bars mean more shares changed hands that day.")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    # Stacked area chart with gradient fills for Close, High, Low
    fig.add_trace(
        go.Scatter(
            x=ohlcv_df.index, y=ohlcv_df["High"], name="High",
            line=dict(color=C_EMERALD, width=1),
            fill=None, mode="lines",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv_df.index, y=ohlcv_df["Close"], name="Close",
            line=dict(color=C_BLUE, width=2.5),
            fill="tonexty", fillcolor="rgba(37,99,235,0.08)",
            mode="lines",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ohlcv_df.index, y=ohlcv_df["Low"], name="Low",
            line=dict(color=C_RED, width=1),
            fill="tonexty", fillcolor="rgba(220,38,38,0.05)",
            mode="lines",
        ),
        row=1, col=1,
    )
    vol_colors = [C_EMERALD if c >= o else C_RED for c, o in zip(ohlcv_df["Close"], ohlcv_df["Open"])]
    fig.add_trace(
        go.Bar(x=ohlcv_df.index, y=ohlcv_df["Volume"], marker_color=vol_colors,
               opacity=1.0, name="Volume", showlegend=False),
        row=2, col=1,
    )
    _apply_layout(fig, height=520, title_text=f"{ticker} — Price & Volume",
                  xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(
        title_text="Volume", row=2, col=1,
        tickfont=dict(color=C_TEXT, size=11),
        title_font=dict(color=C_TEXT, size=13),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Fundamentals ---
    if fund_df is not None and not fund_df.empty:
        _info_header("📋 Fundamental Data",
                     "Quarterly financial data from <strong>SEC EDGAR filings</strong>. "
                     "Revenue = total income from sales; Net Income = profit after all expenses; "
                     "EPS = earnings per share, a key profitability metric for investors.")
        display_cols = [c for c in ["end_date", "Revenue", "Net_Income", "EPS"] if c in fund_df.columns]
        if display_cols:
            fund_display = fund_df[display_cols].copy()
            # Format large numbers
            for col in ["Revenue", "Net_Income"]:
                if col in fund_display.columns:
                    fund_display[col] = fund_display[col].apply(
                        lambda v: f"${v / 1e9:.2f}B" if pd.notna(v) and abs(v) >= 1e9
                        else (f"${v / 1e6:.1f}M" if pd.notna(v) else "N/A")
                    )
            st.dataframe(fund_display, use_container_width=True, hide_index=True)

            # Metric cards for latest
            last = fund_df.iloc[-1]
            fc1, fc2, fc3 = st.columns(3)
            if "Revenue" in fund_df.columns and pd.notna(last.get("Revenue")):
                fc1.metric("Revenue", f"${last['Revenue']/1e9:.2f}B",
                           help="Total money the company earned from selling goods/services in the most recent reported quarter.")
            if "Net_Income" in fund_df.columns and pd.notna(last.get("Net_Income")):
                fc2.metric("Net Income", f"${last['Net_Income']/1e9:.2f}B",
                           help="Profit remaining after subtracting all costs, taxes, and expenses from revenue. Negative = the company lost money.")
            if "EPS" in fund_df.columns and pd.notna(last.get("EPS")):
                fc3.metric("EPS", f"${last['EPS']:.2f}",
                           help="Earnings Per Share — the company's net income divided by shares outstanding. Higher EPS generally signals greater profitability.")
    else:
        st.info("No fundamental data available for this ticker (common for ETFs/indices).")


# ═══════════════════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════════════
def render_technical(df: pd.DataFrame, ticker: str):
    """Tab 2 — technical indicator charts."""

    st.markdown(f"### {ticker} — Technical Analysis")

    # --- Quick metric cards ---
    mc1, mc2, mc3, mc4 = st.columns(4)
    latest = df.iloc[-1]
    if "RSI_14" in df.columns:
        rsi_val = latest["RSI_14"]
        rsi_label = "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral")
        mc1.metric("RSI (14)", f"{rsi_val:.1f}", rsi_label,
                   help="Relative Strength Index — measures momentum on a 0–100 scale. Above 70 = overbought (may drop), below 30 = oversold (may rise), 30–70 = neutral territory.")
    if "MACD" in df.columns and "Signal_Line" in df.columns:
        macd_diff = latest["MACD"] - latest["Signal_Line"]
        mc2.metric("MACD Signal", f"{macd_diff:+.3f}", "Bullish" if macd_diff > 0 else "Bearish",
                   help="MACD minus its Signal Line. Positive = bullish momentum (MACD above Signal), negative = bearish momentum. Crossovers often signal trend changes.")
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        bb_pos = (latest["Close"] - latest["BB_Lower"]) / (latest["BB_Upper"] - latest["BB_Lower"]) * 100 if (latest["BB_Upper"] - latest["BB_Lower"]) != 0 else 50
        mc3.metric("BB Position", f"{bb_pos:.0f}%",
                   help="Where the current price sits within the Bollinger Bands (0%=lower band, 100%=upper band). Near 100% suggests the price is stretched high; near 0% suggests it's stretched low.")
    if "Daily_Returns" in df.columns:
        mc4.metric("Daily Return", f"{latest['Daily_Returns']*100:.2f}%",
                   help="The percentage change in closing price from yesterday to today. This is the raw single-day return for the stock.")

    st.markdown("")

    # --- Moving averages chart ---
    _info_header("📈 Price & Moving Averages",
                 "<strong>Moving averages</strong> smooth out price noise to reveal trends. "
                 "<strong>SMA</strong> (Simple) gives equal weight to all days; <strong>EMA</strong> (Exponential) gives more weight to recent prices. "
                 "When a shorter MA crosses above a longer one, it's a bullish signal (Golden Cross); below = bearish (Death Cross).")
    ma_sel = st.multiselect(
        "Overlays", ["SMA_20", "SMA_50", "EMA_20", "EMA_50"],
        default=["SMA_20", "SMA_50"],
        label_visibility="collapsed",
    )
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color=C_TEXT_DIM, width=1.8, shape="spline"),
        fill="tozeroy", fillcolor="rgba(100,116,139,0.04)",
    ))
    ma_colors = {"SMA_20": C_BLUE, "SMA_50": C_PURPLE, "EMA_20": C_EMERALD, "EMA_50": C_ORANGE}
    for ma in ma_sel:
        if ma in df.columns:
            fig_ma.add_trace(go.Scatter(
                x=df.index, y=df[ma], name=ma,
                line=dict(color=ma_colors.get(ma, "#94a3b8"), width=2, shape="spline"),
                mode="lines+markers", marker=dict(size=3),
            ))
    _apply_layout(fig_ma, height=380, title_text="")
    st.plotly_chart(fig_ma, use_container_width=True)

    # --- RSI ---
    if "RSI_14" in df.columns:
        _info_header("📊 RSI (14)",
                     "<strong>Relative Strength Index</strong> — a momentum oscillator ranging from 0 to 100. "
                     "Measures the speed and magnitude of recent price changes. "
                     "Above <strong>70</strong> = overbought (potential sell signal). "
                     "Below <strong>30</strong> = oversold (potential buy signal). "
                     "The shaded zone (30–70) is neutral territory.")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI",
                                      line=dict(color=C_BLUE, width=2, shape="spline"),
                                      mode="lines+markers", marker=dict(size=2.5)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=C_RED, opacity=0.5, annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=C_EMERALD, opacity=0.5, annotation_text="Oversold")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(37,99,235,0.03)", opacity=1, line_width=0)
        _apply_layout(fig_rsi, height=250, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)

    # --- MACD ---
    if all(c in df.columns for c in ["MACD", "Signal_Line", "MACD_Hist"]):
        _info_header("📉 MACD",
                     "<strong>Moving Average Convergence Divergence</strong> — a trend-following momentum indicator. "
                     "The <strong>MACD line</strong> (blue) = 12-day EMA minus 26-day EMA. "
                     "The <strong>Signal line</strong> (amber, dashed) = 9-day EMA of MACD. "
                     "The <strong>histogram</strong> shows the gap between them — green bars = bullish momentum, red = bearish. "
                     "Buy/sell signals occur when the MACD crosses the Signal line.")
        fig_macd = go.Figure()
        hist_colors = [C_EMERALD if v >= 0 else C_RED for v in df["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram",
                                   marker_color=hist_colors, opacity=0.3))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                       line=dict(color=C_BLUE, width=2, shape="spline"),
                                       mode="lines+markers", marker=dict(size=2.5)))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal_Line"], name="Signal",
                                       line=dict(color=C_ORANGE, width=1.8, dash="dot", shape="spline"),
                                       mode="lines+markers", marker=dict(size=2)))
        _apply_layout(fig_macd, height=280)
        st.plotly_chart(fig_macd, use_container_width=True)

    # --- Bollinger Bands ---
    if all(c in df.columns for c in ["BB_Upper", "BB_Middle", "BB_Lower"]):
        _info_header("🎯 Bollinger Bands",
                     "<strong>Bollinger Bands</strong> show a 20-day moving average with upper and lower bands at ±2 standard deviations. "
                     "Bands widen during volatile periods and narrow during calm ones. "
                     "Price touching the <strong>upper band</strong> may signal overbought conditions; "
                     "touching the <strong>lower band</strong> may signal oversold. "
                     "About 95% of price action occurs within the bands.")
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper",
                                     line=dict(color=C_PURPLE, width=1.2, dash="dash", shape="spline")))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower",
                                     line=dict(color=C_PURPLE, width=1.2, dash="dash", shape="spline"),
                                     fill="tonexty", fillcolor="rgba(124,58,237,0.05)"))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["BB_Middle"], name="Middle",
                                     line=dict(color=C_ORANGE, width=1.2, shape="spline")))
        fig_bb.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                     line=dict(color=C_BLUE, width=2, shape="spline"),
                                     mode="lines+markers", marker=dict(size=2.5)))
        _apply_layout(fig_bb, height=380)
        st.plotly_chart(fig_bb, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# AI PREDICTION TAB
# ═══════════════════════════════════════════════════════════════════════════
def render_prediction(history: dict, probabilities: np.ndarray, actuals: np.ndarray,
                      signal_threshold: float, ticker: str):
    """Tab 3 — training curves, prediction output, accuracy."""

    st.markdown(f"### {ticker} — AI Prediction (LSTM)")

    # --- Training loss curve ---
    _info_header("📉 Training Loss Curve",
                 "Shows how the model's prediction error (BCE loss) decreased during training. "
                 "<strong>Train Loss</strong> = error on training data; <strong>Val Loss</strong> = error on held-out validation data. "
                 "If val loss rises while train loss falls, the model is <strong>overfitting</strong> (memorizing rather than learning patterns).")
    fig_loss = go.Figure()
    epochs_x = list(range(1, len(history["train_losses"]) + 1))
    fig_loss.add_trace(go.Scatter(
        x=epochs_x, y=history["train_losses"], name="Train Loss",
        line=dict(color=C_BLUE, width=2.5, shape="spline"),
        mode="lines+markers", marker=dict(size=5),
    ))
    if history.get("val_losses") and any(v is not None for v in history["val_losses"]):
        val_losses = [v for v in history["val_losses"] if v is not None]
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(val_losses) + 1)), y=val_losses, name="Val Loss",
            line=dict(color=C_ORANGE, width=2, dash="dash", shape="spline"),
            mode="lines+markers", marker=dict(size=5),
        ))
    _apply_layout(fig_loss, height=320, xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown("---")

    # --- Prediction metrics ---
    predictions_binary = (probabilities >= signal_threshold).astype(int)
    accuracy = (predictions_binary == actuals).mean() * 100
    mean_prob = probabilities.mean()
    latest_prob = float(probabilities[-1]) if len(probabilities) > 0 else 0.5
    direction = "UP 🟢" if latest_prob >= signal_threshold else "DOWN 🔴"

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        _info_header("🎯 Latest Signal",
                     "The model's prediction for the most recent day. "
                     "<strong>UP 🟢</strong> means the model predicts the price will rise by more than the threshold over the next N trading days. "
                     "<strong>DOWN 🔴</strong> means it does not. The probability shows the model's confidence level.")
        st.markdown(
            f"<div style='text-align:center; padding:24px; background:{C_SURFACE}; "
            f"border-radius:16px; border:1px solid {C_BORDER}; box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
            f"<span style='font-size:3rem; font-weight:800;'>{direction}</span><br>"
            f"<span style='color:{C_TEXT_DIM}; font-size:0.9rem;'>Probability: {latest_prob:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        _info_header("📊 Test Accuracy",
                     "Percentage of correct predictions on unseen test data (data the model never saw during training). "
                     "Above 55% is generally considered meaningful for financial prediction. "
                     "50% = random guessing; 100% = perfect prediction (unrealistic in markets).")
        acc_color = C_EMERALD if accuracy >= 55 else (C_AMBER if accuracy >= 50 else C_RED)
        # Radial gauge for accuracy
        radius = 60
        circumference = 2 * 3.14159 * radius
        pct = accuracy / 100.0
        dash_val = circumference * pct
        gap_val = circumference * (1 - pct)
        st.markdown(
            f"<div style='text-align:center; padding:24px; background:{C_SURFACE}; "
            f"border-radius:16px; border:1px solid {C_BORDER}; box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
            f"<div class='radial-gauge'>"
            f"<svg width='140' height='140' viewBox='0 0 140 140'>"
            f"<circle cx='70' cy='70' r='{radius}' fill='none' stroke='#e2e8f0' stroke-width='10'/>"
            f"<circle cx='70' cy='70' r='{radius}' fill='none' stroke='{acc_color}' stroke-width='10' "
            f"stroke-dasharray='{dash_val:.1f} {gap_val:.1f}' stroke-linecap='round'/>"
            f"</svg>"
            f"<div class='gauge-label'>"
            f"<span style='font-size:1.6rem; font-weight:800; color:{acc_color};'>{accuracy:.1f}%</span><br>"
            f"<span style='font-size:0.7rem; color:{C_TEXT_DIM};'>ACCURACY</span>"
            f"</div></div>"
            f"<span style='color:{C_TEXT_DIM}; font-size:0.85rem;'>{len(actuals)} test samples</span></div>",
            unsafe_allow_html=True,
        )

    with col_c:
        _info_header("📈 Avg Probability",
                     "The average predicted probability of a price increase across all test samples. "
                     "Values near 50% suggest the model is uncertain; values strongly above or below 50% show directional conviction.")
        st.markdown(
            f"<div style='text-align:center; padding:24px; background:{C_SURFACE}; "
            f"border-radius:16px; border:1px solid {C_BORDER}; box-shadow:0 1px 3px rgba(0,0,0,0.04);'>"
            f"<span style='font-size:3rem; font-weight:800; color:{C_PURPLE};'>"
            f"{mean_prob:.1%}</span><br>"
            f"<span style='color:{C_TEXT_DIM}; font-size:0.9rem;'>Mean P(Up)</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Probability distribution ---
    _info_header("🔬 Prediction Probability Distribution",
                 "Histogram showing how the model's predicted probabilities are distributed across all test samples. "
                 "The <strong>amber dashed line</strong> marks the signal threshold. "
                 "Samples to the right trigger BUY signals; to the left trigger HOLD/SELL. "
                 "A well-calibrated model shows a bimodal distribution (clusters near 0 and 1).")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=probabilities, nbinsx=40, name="P(Up)",
        marker_color=C_BLUE, opacity=0.6,
    ))
    fig_hist.add_vline(x=signal_threshold, line_dash="dash", line_color=C_ORANGE,
                        annotation_text=f"Threshold ({signal_threshold})")
    _apply_layout(fig_hist, height=280, xaxis_title="Probability", yaxis_title="Count")
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Explanation ---
    with st.expander("ℹ️ How it works"):
        st.markdown(f"""
        The LSTM model processes sequences of **{len(FEATURE_COLUMNS)} features** over a sliding window to predict
        whether the stock price will increase above a threshold over the next trading days.

        **Features used:**
        `{', '.join(FEATURE_COLUMNS)}`

        A probability ≥ **{signal_threshold:.0%}** triggers a **BUY** signal; otherwise the model suggests holding cash.
        """)


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST TAB
# ═══════════════════════════════════════════════════════════════════════════
def render_backtest(portfolio_df: pd.DataFrame, metrics: dict, initial_capital: float, ticker: str):
    """Tab 4 — equity curve, drawdown, performance cards, trade log."""

    st.markdown(f"### {ticker} — Backtest Results")

    # --- Metric cards ---
    cols = st.columns(6)
    metric_keys = [
        ("Total Return", "Total_Return_Pct", "%", C_EMERALD),
        ("Ann. Return", "Annualized_Return_Pct", "%", C_BLUE),
        ("Sharpe Ratio", "Sharpe_Ratio", "", C_PURPLE),
        ("Max Drawdown", "Max_Drawdown_Pct", "%", C_RED),
        ("Win Rate", "Win_Rate_Pct", "%", C_AMBER),
        ("Total Trades", "Total_Trades", "", C_TEXT),
    ]
    _metric_help = {
        "Total_Return_Pct": "Total percentage gain or loss from the strategy over the entire backtest period. Positive = profit, negative = loss.",
        "Annualized_Return_Pct": "The total return scaled to a yearly rate, making it comparable across different time periods. Uses 252 trading days per year.",
        "Sharpe_Ratio": "Risk-adjusted return — measures excess return per unit of risk (volatility). Above 1.0 is good, above 2.0 is very good, below 0 means the strategy lost money vs. risk-free rate.",
        "Max_Drawdown_Pct": "The largest peak-to-trough decline in portfolio value. Shows the worst-case loss you would have experienced. Lower is better.",
        "Win_Rate_Pct": "Percentage of completed round-trip trades (buy → sell) that were profitable. Above 50% means more winning trades than losing ones.",
        "Total_Trades": "Total number of completed buy→sell round-trip trades executed during the backtest period.",
    }
    for col, (label, key, suffix, _color) in zip(cols, metric_keys):
        val = metrics.get(key, "N/A")
        fmt = f"{val:.2f}{suffix}" if isinstance(val, (int, float)) else str(val)
        col.metric(label, fmt, help=_metric_help.get(key, ""))

    st.markdown("")

    # --- Equity curve vs buy-and-hold ---
    _info_header("💰 Portfolio Equity vs Buy & Hold",
                 "Compares the <strong>AI strategy's</strong> portfolio value (blue) against a simple <strong>Buy & Hold</strong> approach (amber dashed). "
                 "Buy & Hold assumes you invested all capital at the start and never traded. "
                 "The dotted line marks your initial capital. If blue stays above amber, the AI strategy is outperforming.")
    if "Portfolio_Value" in portfolio_df.columns and "Price" in portfolio_df.columns:
        date_col = portfolio_df["Date"] if "Date" in portfolio_df.columns else portfolio_df.index
        bh_shares = initial_capital / portfolio_df["Price"].iloc[0]
        bh_value = bh_shares * portfolio_df["Price"]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=date_col, y=portfolio_df["Portfolio_Value"], name="Strategy",
            line=dict(color=C_BLUE, width=2.5, shape="spline"),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        ))
        fig_eq.add_trace(go.Scatter(
            x=date_col, y=bh_value, name="Buy & Hold",
            line=dict(color=C_ORANGE, width=1.8, dash="dash", shape="spline"),
        ))
        fig_eq.add_hline(y=initial_capital, line_dash="dot", line_color=C_TEXT_DIM, opacity=0.3)
        _apply_layout(fig_eq, height=400, yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_eq, use_container_width=True)

    # --- Drawdown ---
    _info_header("📉 Drawdown",
                 "Shows the percentage drop from the portfolio's all-time high at each point in time. "
                 "A drawdown of -10% means the portfolio was 10% below its peak. "
                 "Deeper/longer drawdowns indicate higher risk. The worst point is the <strong>Maximum Drawdown</strong>.")
    if "Portfolio_Value" in portfolio_df.columns:
        cum_max = portfolio_df["Portfolio_Value"].cummax()
        drawdown = (portfolio_df["Portfolio_Value"] - cum_max) / cum_max * 100
        date_col = portfolio_df["Date"] if "Date" in portfolio_df.columns else portfolio_df.index

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=date_col, y=drawdown, name="Drawdown",
            line=dict(color=C_RED, width=1.8, shape="spline"),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.07)",
        ))
        _apply_layout(fig_dd, height=250, yaxis_title="Drawdown (%)")
        st.plotly_chart(fig_dd, use_container_width=True)

    # --- Buy & Hold return comparison ---
    bh_ret = metrics.get("Buy_Hold_Return_Pct")
    strat_ret = metrics.get("Total_Return_Pct")
    if bh_ret is not None and strat_ret is not None:
        diff = strat_ret - bh_ret
        sign = "outperformed" if diff > 0 else "underperformed"
        color = C_EMERALD if diff > 0 else C_RED
        st.markdown(
            f"<div style='text-align:center; padding:14px; background:{C_SURFACE}; "
            f"border-radius:12px; border:1px solid {C_BORDER}; margin-bottom:16px;'>"
            f"Strategy <b style=\"color:{color}\">{sign}</b> Buy & Hold by "
            f"<b style=\"color:{color}\">{abs(diff):.2f}%</b></div>",
            unsafe_allow_html=True,
        )

    # --- Trade log ---
    _info_header("📜 Trade Log",
                 "Detailed log of every BUY and SELL action taken by the strategy. "
                 "Shows the date, model signal, price at execution, shares held, cash remaining, and total portfolio value at each trade.")
    display_cols = [c for c in ["Date", "Signal", "Price", "Action", "Shares_Held",
                                 "Capital", "Portfolio_Value"] if c in portfolio_df.columns]
    trades = portfolio_df[portfolio_df["Action"] != "HOLD"] if "Action" in portfolio_df.columns else portfolio_df
    if len(trades) > 0:
        st.dataframe(
            trades[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=320,
        )
    else:
        st.info("No trades were executed during the backtest period.")


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# STOCK SCREENER TAB
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=7200)
def _fetch_sp500_tickers() -> pd.DataFrame:
    """Fetch the current S&P 500 list from Wikipedia."""
    import requests
    from io import StringIO
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    tables = pd.read_html(StringIO(response.text))
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["Ticker", "Company", "Sector", "Sub_Industry"]
    # Fix tickers with dots (BRK.B -> BRK-B for yfinance)
    df["Ticker"] = df["Ticker"].str.replace(".", "-", regex=False)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def _batch_download(tickers_str: str, period: str = "6mo") -> pd.DataFrame:
    """Batch download OHLCV data for multiple tickers."""
    import yfinance
    data = yfinance.download(tickers_str, period=period, progress=False, timeout=60, threads=True)
    return data


def _score_stock(closes: pd.Series) -> dict:
    """Compute a composite buy score (0-100) from technical signals for a single stock."""
    if closes is None or len(closes) < 60:
        return None

    closes = closes.dropna()
    if len(closes) < 60:
        return None

    price = closes.iloc[-1]
    scores = {}

    # --- RSI (14) ---
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])
    # RSI scoring: oversold = buy opportunity
    if rsi_val < 25:
        scores["RSI"] = 10
    elif rsi_val < 35:
        scores["RSI"] = 8
    elif rsi_val < 45:
        scores["RSI"] = 6
    elif rsi_val < 55:
        scores["RSI"] = 5
    elif rsi_val < 65:
        scores["RSI"] = 4
    elif rsi_val < 75:
        scores["RSI"] = 2
    else:
        scores["RSI"] = 0

    # --- MACD crossover ---
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_val = float(macd.iloc[-1])
    signal_val = float(signal.iloc[-1])
    hist = macd_val - signal_val
    hist_prev = float(macd.iloc[-2] - signal.iloc[-2])
    # Bullish crossover / above signal = positive
    if hist > 0 and hist_prev <= 0:
        scores["MACD"] = 10  # fresh bullish crossover
    elif hist > 0 and hist > hist_prev:
        scores["MACD"] = 8  # bullish and accelerating
    elif hist > 0:
        scores["MACD"] = 6  # bullish
    elif hist < 0 and hist > hist_prev:
        scores["MACD"] = 4  # bearish but improving
    elif hist < 0:
        scores["MACD"] = 1  # bearish
    else:
        scores["MACD"] = 5

    # --- Moving average trend ---
    sma20 = closes.rolling(20).mean().iloc[-1]
    sma50 = closes.rolling(50).mean().iloc[-1]
    above_20 = price > sma20
    above_50 = price > sma50
    sma20_rising = sma20 > closes.rolling(20).mean().iloc[-5]
    if above_20 and above_50 and sma20_rising:
        scores["Trend"] = 10
    elif above_20 and above_50:
        scores["Trend"] = 8
    elif above_20:
        scores["Trend"] = 6
    elif above_50:
        scores["Trend"] = 4
    else:
        scores["Trend"] = 1

    # --- Bollinger Band position ---
    bb_mid = closes.rolling(20).mean()
    bb_std = closes.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    if bb_range > 0:
        bb_pct = (price - float(bb_lower.iloc[-1])) / bb_range
    else:
        bb_pct = 0.5
    # Near lower band = buy opportunity
    if bb_pct < 0.15:
        scores["BB"] = 10
    elif bb_pct < 0.3:
        scores["BB"] = 8
    elif bb_pct < 0.5:
        scores["BB"] = 6
    elif bb_pct < 0.7:
        scores["BB"] = 5
    elif bb_pct < 0.85:
        scores["BB"] = 3
    else:
        scores["BB"] = 1

    # --- Momentum (20-day return) ---
    mom_20 = (price / closes.iloc[-20] - 1) * 100 if len(closes) >= 20 else 0
    mom_5 = (price / closes.iloc[-5] - 1) * 100 if len(closes) >= 5 else 0
    # Moderate positive momentum is best (not overextended)
    if 2 < mom_20 < 10 and mom_5 > 0:
        scores["Momentum"] = 10
    elif 0 < mom_20 < 15 and mom_5 > 0:
        scores["Momentum"] = 8
    elif mom_20 > 0:
        scores["Momentum"] = 6
    elif -5 < mom_20 < 0 and mom_5 > 0:
        scores["Momentum"] = 5  # dip-buy opportunity
    elif mom_20 < -10:
        scores["Momentum"] = 2
    else:
        scores["Momentum"] = 3

    # --- Composite score (weighted average out of 100) ---
    weights = {"RSI": 0.20, "MACD": 0.25, "Trend": 0.25, "BB": 0.15, "Momentum": 0.15}
    composite = sum(scores[k] * weights[k] for k in scores) * 10  # scale to 0-100
    composite = max(0.0, min(100.0, composite))

    return {
        "Price": round(price, 2),
        "RSI": round(rsi_val, 1),
        "MACD_Hist": round(hist, 4),
        "SMA20_Dist": round((price / sma20 - 1) * 100, 2),
        "BB_Pct": round(bb_pct * 100, 1),
        "Mom_20d": round(mom_20, 2),
        "Mom_5d": round(mom_5, 2),
        "Score": round(composite, 1),
        "Signal": "🟢 Strong Buy" if composite >= 75 else (
            "🟢 Buy" if composite >= 60 else (
                "🟡 Hold" if composite >= 40 else (
                    "🔴 Sell" if composite >= 25 else "🔴 Strong Sell"
                )
            )
        ),
        **{f"s_{k}": v for k, v in scores.items()},
    }


def _ai_score_stock(ticker: str, raw_data: pd.DataFrame, model: torch.nn.Module, seq_len: int) -> dict:
    """Compute buy score using the trained LSTM model."""
    if not isinstance(raw_data.columns, pd.MultiIndex):
        return None
        
    if ticker not in raw_data.columns.levels[1]:
        return None

    ticker_df = pd.DataFrame({
        "Open": raw_data["Open"][ticker],
        "High": raw_data["High"][ticker],
        "Low": raw_data["Low"][ticker],
        "Close": raw_data["Close"][ticker],
        "Volume": raw_data["Volume"][ticker],
    })
    
    ticker_df = ticker_df.dropna()
    if len(ticker_df) < seq_len + 60: # need extra rows for moving averages to compute correctly
        return None

    # Calculate indicators
    try:
        ticker_df = calculate_technical_indicators(ticker_df)
    except Exception:
        return None
        
    ticker_df = ticker_df.dropna()
    if len(ticker_df) < seq_len:
        return None
        
    price = ticker_df["Close"].iloc[-1]
    
    # Scale features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ticker_df[FEATURE_COLUMNS].values)
    
    # Predict using the last seq_len rows
    X = torch.tensor(scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        prob = model(X.to(device)).item()
        
    composite = prob * 100
    
    # Gather technicals for the table
    rsi_val = ticker_df["RSI_14"].iloc[-1]
    sma20 = ticker_df["SMA_20"].iloc[-1]
    macd_val = ticker_df["MACD"].iloc[-1]
    signal_val = ticker_df["Signal_Line"].iloc[-1]
    hist = macd_val - signal_val
    
    bb_upper = ticker_df["BB_Upper"].iloc[-1]
    bb_lower = ticker_df["BB_Lower"].iloc[-1]
    bb_range = bb_upper - bb_lower
    bb_pct = (price - bb_lower) / bb_range if bb_range > 0 else 0.5
    
    mom_20 = (price / ticker_df["Close"].iloc[-20] - 1) * 100 if len(ticker_df) >= 20 else 0
    mom_5 = (price / ticker_df["Close"].iloc[-5] - 1) * 100 if len(ticker_df) >= 5 else 0

    return {
        "Price": round(price, 2),
        "RSI": round(rsi_val, 1),
        "MACD_Hist": round(hist, 4),
        "SMA20_Dist": round((price / sma20 - 1) * 100, 2),
        "BB_Pct": round(bb_pct * 100, 1),
        "Mom_20d": round(mom_20, 2),
        "Mom_5d": round(mom_5, 2),
        "Score": round(composite, 1),
        "Signal": "🟢 Strong Buy" if composite >= 75 else (
            "🟢 Buy" if composite >= 60 else (
                "🟡 Hold" if composite >= 40 else (
                    "🔴 Sell" if composite >= 25 else "🔴 Strong Sell"
                )
            )
        ),
    }


def render_screener():
    """Tab 5 — S&P 500 stock screener with composite buy scores."""

    st.markdown("### 🏆 S&P 500 Stock Screener")
    _info_header("Composite Buy Ranking",
                 "Ranks all S&P 500 stocks by a <strong>buy score</strong> (0–100). "
                 "You can score stocks using the hardcoded <strong>Technical Composite</strong> "
                 "(RSI, MACD, Moving Averages, etc.) OR the <strong>AI Model</strong> trained on the currently active stock.")

    st.markdown("")
    
    scoring_mode = st.radio("Scoring Method", ["Technical Composite", "AI Model (LSTM)"], horizontal=True)

    # --- Controls ---
    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1])
    with ctrl1:
        top_n = st.select_slider("Show top N stocks", options=[10, 25, 50, 100, 200, 500], value=50)
    with ctrl2:
        sector_filter = st.selectbox("Filter by sector", ["All Sectors"] + sorted([
            "Information Technology", "Health Care", "Financials", "Consumer Discretionary",
            "Communication Services", "Industrials", "Consumer Staples", "Energy",
            "Utilities", "Real Estate", "Materials",
        ]))
    with ctrl3:
        sort_col = st.selectbox("Sort by", ["Score", "RSI", "Mom_20d", "Mom_5d", "Price"], index=0)

    run_scan = st.button("🔍 Scan S&P 500", use_container_width=True, type="primary")

    # Use session state to persist scan results
    if "screener_results" not in st.session_state:
        st.session_state.screener_results = None

    if run_scan:
        if scoring_mode == "AI Model (LSTM)":
            if st.session_state.results is None or "model" not in st.session_state.results:
                st.error("Please run the Analysis Pipeline (sidebar) first to train the AI model.")
                return
        # Step 1: Fetch S&P 500 list
        with st.spinner("📋 Fetching S&P 500 constituent list …"):
            try:
                sp500 = _fetch_sp500_tickers()
                st.success(f"✅ Loaded {len(sp500)} S&P 500 companies")
            except Exception as e:
                st.error(f"Failed to fetch S&P 500 list: {e}")
                return

        # Step 2: Batch download price data
        tickers_list = sp500["Ticker"].tolist()
        tickers_str = " ".join(tickers_list)

        with st.spinner("📥 Downloading price data for 500 stocks (this may take 30-60 seconds) …"):
            try:
                raw = _batch_download(tickers_str, period="6mo")
            except Exception as e:
                st.error(f"Download failed: {e}")
                return

        if raw.empty:
            st.error("No data received from Yahoo Finance. Try again in a few minutes.")
            return

        # Extract Close prices (handle MultiIndex columns) for Technical mode
        if isinstance(raw.columns, pd.MultiIndex):
            close_data = raw["Close"]
            all_tickers = raw.columns.levels[1].tolist()
        else:
            close_data = raw[["Close"]]
            all_tickers = raw.columns.tolist()

        # Step 3: Score each stock
        progress = st.progress(0, text="📊 Scoring stocks …")
        scored = []
        valid_tickers = [t for t in tickers_list if t in all_tickers]
        
        model = None
        seq_len = 60
        if scoring_mode == "AI Model (LSTM)":
            model = st.session_state.results["model"]
            seq_len = st.session_state.last_params.get("sequence_length", 60)

        for i, ticker in enumerate(valid_tickers):
            if i % 25 == 0:
                progress.progress(
                    min(i / max(len(valid_tickers), 1), 1.0),
                    text=f"📊 Scoring {ticker} ({i+1}/{len(valid_tickers)}) …"
                )
            try:
                if scoring_mode == "AI Model (LSTM)":
                    result = _ai_score_stock(ticker, raw, model, seq_len)
                else:
                    if ticker in close_data.columns:
                        result = _score_stock(close_data[ticker])
                    else:
                        result = None
                        
                if result is not None:
                    info = sp500[sp500["Ticker"] == ticker].iloc[0]
                    result["Ticker"] = ticker
                    result["Company"] = info["Company"]
                    result["Sector"] = info["Sector"]
                    scored.append(result)
            except Exception as e:
                continue

        progress.progress(1.0, text=f"✅ Scored {len(scored)} stocks")

        if not scored:
            st.error("No stocks could be scored. Try again later.")
            return

        results_df = pd.DataFrame(scored)
        st.session_state.screener_results = results_df

    # --- Display results ---
    results_df = st.session_state.screener_results
    if results_df is None:
        st.markdown(
            f"<div style='text-align:center; padding:60px 20px; color:{C_TEXT_DIM};'>"
            f"<span style='font-size:3.5rem;'>🏆</span><br><br>"
            f"<span style='font-size:1.1rem;'>Click <b style=\"color:{C_BLUE};\">Scan S&P 500</b> "
            f"to rank all 500 stocks by buy potential.</span><br>"
            f"<span style='font-size:0.85rem; color:{C_TEXT_DIM};'>This uses RSI, MACD, Moving Averages, "
            f"Bollinger Bands, and Momentum to compute a composite score.</span></div>",
            unsafe_allow_html=True,
        )
        return

    # Apply filters
    filtered = results_df.copy()
    if sector_filter != "All Sectors":
        filtered = filtered[filtered["Sector"] == sector_filter]

    ascending = sort_col == "RSI"  # lower RSI = more oversold = better buy
    filtered = filtered.sort_values(sort_col, ascending=ascending).head(top_n).reset_index(drop=True)
    filtered.index = filtered.index + 1  # 1-based ranking

    # --- Top 5 picks ---
    top5 = filtered.head(5)
    if len(top5) > 0:
        st.markdown("#### 🌟 Top Picks")
        pick_cols = st.columns(min(5, len(top5)))
        for col, (_, row) in zip(pick_cols, top5.iterrows()):
            score = row["Score"]
            score_color = C_EMERALD if score >= 70 else (C_BLUE if score >= 55 else (C_AMBER if score >= 40 else C_RED))
            col.markdown(
                f"<div style='text-align:center; padding:20px 12px; background:{C_SURFACE}; "
                f"border-radius:12px; border:1px solid {C_BORDER}; "
                f"box-shadow:0 1px 4px rgba(0,0,0,0.05); transition: transform 0.2s ease; cursor:default;'>"
                f"<div style='font-size:0.65rem; color:{C_TEXT_DIM}; text-transform:uppercase; "
                f"letter-spacing:0.07em; margin-bottom:6px;'>{row['Sector'][:20]}</div>"
                f"<div style='font-size:1.4rem; font-weight:800; color:{C_TEXT};'>{row['Ticker']}</div>"
                f"<div style='font-size:0.78rem; color:{C_TEXT_DIM}; margin:2px 0 10px; "
                f"white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{row['Company'][:22]}</div>"
                f"<div style='font-size:2rem; font-weight:800; color:{score_color};'>{score:.0f}</div>"
                f"<div style='font-size:0.68rem; color:{C_TEXT_DIM}; text-transform:uppercase; letter-spacing:0.06em;'>BUY SCORE</div>"
                f"<div style='margin-top:8px; font-size:0.78rem;'>{row['Signal']}</div>"
                f"<div style='margin-top:6px; font-size:0.75rem; color:{C_TEXT_DIM};'>"
                f"${row['Price']:.2f} · RSI {row['RSI']:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("")

    # --- Sector breakdown chart ---
    sector_avg = filtered.groupby("Sector")["Score"].mean().sort_values(ascending=True)
    if len(sector_avg) > 1:
        _info_header("📊 Average Score by Sector",
                     "Average composite buy score for each GICS sector in the filtered results. "
                     "Higher scores indicate sectors with more technically attractive stocks right now.")
        fig_sector = go.Figure()
        bar_colors = [C_EMERALD if v >= 60 else (C_BLUE if v >= 50 else (C_AMBER if v >= 40 else C_RED))
                      for v in sector_avg.values]
        fig_sector.add_trace(go.Bar(
            y=sector_avg.index,
            x=sector_avg.values,
            orientation="h",
            marker_color=bar_colors,
            opacity=0.85,
            text=[f"{v:.1f}" for v in sector_avg.values],
            textposition="outside",
            textfont=dict(size=11, color=C_TEXT),
        ))
        _apply_layout(fig_sector, height=max(250, len(sector_avg) * 38),
                      xaxis_title="Avg Buy Score", yaxis_title="",
                      margin=dict(l=160, r=50, t=30, b=40))
        st.plotly_chart(fig_sector, use_container_width=True)

    # --- Full ranking table ---
    _info_header("📋 Full Ranking Table",
                 "All scored stocks sorted by the selected criteria. "
                 "<strong>Score</strong> = composite buy signal (0–100). "
                 "<strong>RSI</strong> = momentum (below 30 = oversold). "
                 "<strong>SMA20 Dist</strong> = % above/below 20-day average. "
                 "<strong>BB%</strong> = position in Bollinger Band (low = near support). "
                 "<strong>Mom 20d/5d</strong> = 20/5-day price momentum.")

    display_df = filtered[[
        "Ticker", "Company", "Sector", "Signal", "Score",
        "Price", "RSI", "MACD_Hist", "SMA20_Dist", "BB_Pct", "Mom_20d", "Mom_5d",
    ]].copy()
    display_df.columns = [
        "Ticker", "Company", "Sector", "Signal", "Score",
        "Price ($)", "RSI", "MACD Hist", "SMA20 Dist %", "BB %", "Mom 20d %", "Mom 5d %",
    ]

    # Color the Score column
    def _color_score(val):
        if val >= 70:
            return f"color: {C_EMERALD}; font-weight: 700;"
        elif val >= 55:
            return f"color: {C_BLUE}; font-weight: 600;"
        elif val >= 40:
            return f"color: {C_AMBER};"
        else:
            return f"color: {C_RED};"

    styled = display_df.style.map(
        _color_score, subset=["Score"]
    ).format({
        "Price ($)": "${:.2f}",
        "RSI": "{:.1f}",
        "MACD Hist": "{:.4f}",
        "SMA20 Dist %": "{:+.2f}%",
        "BB %": "{:.1f}%",
        "Mom 20d %": "{:+.2f}%",
        "Mom 5d %": "{:+.2f}%",
        "Score": "{:.1f}",
    })

    st.dataframe(styled, use_container_width=True, height=min(700, 40 + len(display_df) * 36))

    # --- Score distribution ---
    _info_header("📊 Score Distribution",
                 "Histogram showing how buy scores are distributed across the scanned stocks. "
                 "A right-skewed distribution means most stocks are in buy territory; "
                 "left-skewed means most are in sell/hold territory.")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=results_df["Score"], nbinsx=30, name="Score Distribution",
        marker_color=C_BLUE, opacity=0.6,
    ))
    fig_dist.add_vline(x=60, line_dash="dash", line_color=C_EMERALD, opacity=0.6,
                        annotation_text="Buy Zone")
    fig_dist.add_vline(x=40, line_dash="dash", line_color=C_RED, opacity=0.6,
                        annotation_text="Sell Zone")
    _apply_layout(fig_dist, height=280, xaxis_title="Buy Score", yaxis_title="# Stocks")
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Summary stats ---
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    total = len(results_df)
    strong_buy = len(results_df[results_df["Score"] >= 75])
    buy = len(results_df[(results_df["Score"] >= 60) & (results_df["Score"] < 75)])
    hold = len(results_df[(results_df["Score"] >= 40) & (results_df["Score"] < 60)])
    sell = len(results_df[results_df["Score"] < 40])
    sc1.metric("Stocks Scanned", total, help="Total S&P 500 stocks successfully scanned and scored.")
    sc2.metric("🟢 Strong Buy", strong_buy, help="Stocks with score ≥ 75 — strong technical buy signals across multiple indicators.")
    sc3.metric("🟢 Buy", buy, help="Stocks with score 60–74 — moderately bullish technical picture.")
    sc4.metric("🟡 Hold", hold, help="Stocks with score 40–59 — mixed signals, no clear directional edge.")
    sc5.metric("🔴 Sell", sell, help="Stocks with score < 40 — bearish technical signals, potential downside risk.")


def _log_admin(msg: str, level: str = "success"):
    """Helper to store logs for the Admin Data tab."""
    if "admin_logs" not in st.session_state:
        st.session_state.admin_logs = []
    st.session_state.admin_logs.append({"msg": msg, "level": level})

def render_admin_data():
    """Render the Admin Data tab."""
    st.markdown("### 🛠 Admin Data & Logs")
    logs = st.session_state.get("admin_logs", [])
    if not logs:
        st.info("No logs available. Run the analysis pipeline first to generate logs.")
        return
    for log in logs:
        if log["level"] == "success":
            st.success(log["msg"])
        elif log["level"] == "warning":
            st.warning(log["msg"])
        elif log["level"] == "info":
            st.info(log["msg"])
        else:
            st.error(log["msg"])


def run_pipeline(params: dict) -> dict | None:
    """Execute full analysis pipeline; returns dict of results or None on error."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict = {}
    st.session_state.admin_logs = []  # Clear previous logs

    # 1 — Fetch OHLCV
    with st.spinner("📡 Fetching OHLCV data …"):
        ohlcv_df = _cached_ohlcv(params["ticker"], params["start_date"], params["end_date"])
    if ohlcv_df is None or ohlcv_df.empty:
        st.error("Failed to fetch OHLCV data. Check the ticker and date range.")
        return None
    results["ohlcv"] = ohlcv_df
    msg = f"✅ OHLCV loaded — {len(ohlcv_df)} rows"
    _log_admin(msg, "success")

    # 2 — Fetch fundamentals
    with st.spinner("📋 Fetching fundamental data …"):
        fund_df = _cached_fundamentals(params["ticker"], params["sec_email"])
    results["fund"] = fund_df
    if fund_df is not None:
        _log_admin(f"✅ Fundamentals loaded — {len(fund_df)} records", "success")
    else:
        msg = "⚠ No fundamental data (common for ETFs)."
        _log_admin(msg, "warning")

    # 3 — Merge
    with st.spinner("🔗 Merging data …"):
        try:
            merged = align_and_merge(ohlcv_df, fund_df)
        except Exception as exc:
            st.error(f"Data merge failed: {exc}")
            return None

    # 4 — Indicators
    with st.spinner("📐 Calculating technical indicators …"):
        try:
            df = calculate_technical_indicators(merged)
        except Exception as exc:
            st.error(f"Indicator calculation failed: {exc}")
            return None
    results["df_indicators"] = df

    # 5 — Labels
    with st.spinner("🏷 Creating target labels …"):
        try:
            labels = create_target_labels(
                df,
                horizon=params["prediction_horizon"],
                threshold=params["classification_threshold"],
            )
        except Exception as exc:
            st.error(f"Label creation failed: {exc}")
            return None

    # 6 — Drop NaNs
    df = df.copy()
    df["_target"] = labels
    subset = [c for c in FEATURE_COLUMNS + ["_target"] if c in df.columns]
    df.dropna(subset=subset, inplace=True)

    if len(df) < params["sequence_length"] + 20:
        st.error(f"Not enough data after cleaning ({len(df)} rows). Try a wider date range.")
        return None

    labels_clean = df["_target"].values.astype(np.float32)
    results["df_clean"] = df

    # 7 — Train / test split
    train_end = int(len(df) * 0.8)
    msg = f"Train: {train_end} rows  |  Test: {len(df) - train_end} rows"
    _log_admin(msg, "info")

    # 8 — Scale features
    with st.spinner("⚖ Scaling features …"):
        feat_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        try:
            scaled_train, scaled_test, scaler = scale_features(df, train_end, feat_cols)
        except Exception as exc:
            st.error(f"Scaling failed: {exc}")
            return None

    train_labels = labels_clean[:train_end]
    test_labels = labels_clean[train_end:]
    seq_len = params["sequence_length"]

    # 9 — DataLoaders
    train_ds = StockDataset(scaled_train, train_labels, sequence_length=seq_len)
    test_ds = StockDataset(scaled_test, test_labels, sequence_length=seq_len)

    if len(train_ds) == 0 or len(test_ds) == 0:
        st.error("Datasets empty after sequencing. Need more data or shorter sequence length.")
        return None

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=params["batch_size"], shuffle=False)

    # Optional val split — only if the leftover slice has enough rows
    val_loader = None
    val_split = int(len(scaled_train) * 0.9)
    val_size = len(scaled_train) - val_split
    if val_size > seq_len + 1:
        try:
            val_ds = StockDataset(scaled_train[val_split:], train_labels[val_split:], sequence_length=seq_len)
            if len(val_ds) > 0:
                val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
        except ValueError:
            val_loader = None  # not enough data for validation — skip silently

    # 10 — Train model
    input_size = scaled_train.shape[1]
    model = LSTMModel(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    with st.spinner(f"🧠 Training LSTM ({params['epochs']} epochs) …"):
        try:
            history = train_model(
                model, train_loader, val_loader=val_loader,
                epochs=params["epochs"], lr=params["learning_rate"], device=device,
            )
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return None
    results["history"] = history
    msg = f"✅ Training complete — final loss: {history['train_losses'][-1]:.5f}"
    _log_admin(msg, "success")

    # 10.5 — Train Random Forest Ensemble
    with st.spinner("🌲 Training Random Forest Ensemble …"):
        try:
            rf_model = train_random_forest(train_ds, n_estimators=100)
            results["rf_model"] = rf_model
        except Exception as exc:
            st.error(f"Random Forest Training failed: {exc}")
            return None

    # 11 — Evaluate
    with st.spinner("🔍 Evaluating ensemble model …"):
        try:
            probabilities, actuals = evaluate_model(
                model, test_loader, device=device, rf_model=rf_model, rf_weight=0.5
            )
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return None
    results["probabilities"] = probabilities
    results["actuals"] = actuals
    results["model"] = model

    # 12 — Backtest
    with st.spinner("📈 Running backtest …"):
        test_df = df.iloc[train_end:]
        bt_start = seq_len
        bt_close = test_df["Close"].iloc[bt_start: bt_start + len(probabilities)]
        bt_dates = test_df.index[bt_start: bt_start + len(probabilities)]

        try:
            portfolio_df = run_backtest(
                close_prices=bt_close,
                predictions=probabilities,
                dates=bt_dates,
                initial_capital=params["initial_capital"],
                signal_threshold=params["signal_threshold"],
            )
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")
            return None

        metrics = calculate_performance_metrics(
            portfolio_df,
            initial_capital=params["initial_capital"],
            risk_free_rate=params["risk_free_rate"],
        )

    results["portfolio"] = portfolio_df
    results["metrics"] = metrics
    msg = "✅ Backtest complete!"
    _log_admin(msg, "success")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def app():
    """Entry point for the Streamlit dashboard."""
    load_dotenv()

    # --- Hero header ---
    st.markdown(
        '<p class="hero-title">Stock Market Analyzer</p>'
        '<p class="hero-sub">AI-powered stock prediction & backtesting dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # --- Sidebar ---
    params = render_sidebar()

    # --- Session state ---
    if "results" not in st.session_state:
        st.session_state.results = None

    if params.get("mode") == "Screener":
        render_screener()
        return
    elif params.get("mode") == "Admin Data":
        render_admin_data()
        return

    # --- Run pipeline ---
    if params.get("run"):
        st.session_state.results = run_pipeline(params)
        st.session_state.last_params = params

    # --- Display results ---
    results = st.session_state.results
    if results is None:
        st.markdown(
            f"<div style='text-align:center; padding:80px 20px; color:{C_TEXT_DIM};'>"
            f"<span style='font-size:4rem;'>📊</span><br><br>"
            f"<span style='font-size:1.2rem;'>Configure parameters in the sidebar and click "
            f"<b style=\"color:{C_BLUE};\">Run Analysis</b> to get started.</span></div>",
            unsafe_allow_html=True,
        )
        return

    ticker = params.get("ticker", st.session_state.get("last_params", {}).get("ticker", ""))

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔬 Technical Analysis",
        "🤖 AI Prediction",
        "📈 Backtest Results",
    ])

    with tab1:
        render_overview(results["ohlcv"], results.get("fund"), ticker)

    with tab2:
        render_technical(results.get("df_indicators", results.get("df_clean", pd.DataFrame())), ticker)

    with tab3:
        if "history" in results and "probabilities" in results:
            render_prediction(
                results["history"], results["probabilities"], results["actuals"],
                params.get("signal_threshold", st.session_state.get("last_params", {}).get("signal_threshold", 0.5)),
                ticker,
            )
        else:
            st.warning("Model results not available.")

    with tab4:
        if "portfolio" in results and "metrics" in results:
            render_backtest(
                results["portfolio"], results["metrics"],
                params.get("initial_capital", st.session_state.get("last_params", {}).get("initial_capital", 10000)),
                ticker,
            )
        else:
            st.warning("Backtest results not available.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
