import datetime as dt
import streamlit as st
import pandas as pd

from market import (
    MarketConfig,
    DataMode,
    EquityMarketData,
    EquityConfig,
    OptionChainMarketData,
    OptionChainConfig,
)


st.set_page_config(page_title="Market Data", layout="wide")

st.title("📈 Market Data – Equity & Options")

# --- Config globale (partagée avec app.py) ---
if "market_config" not in st.session_state:
    st.session_state["market_config"] = MarketConfig(
        valuation_date=dt.date.today(),
        mode=DataMode.SNAPSHOT,
        currency="USD",
        data_dir="data",
    )

cfg: MarketConfig = st.session_state["market_config"]

col1, col2, col3 = st.columns(3)
with col1:
    new_date = st.date_input("Valuation date", value=cfg.valuation_date)
with col2:
    mode_str = st.selectbox("Data mode", options=[DataMode.SNAPSHOT.value, DataMode.LIVE.value], index=0)
    new_mode = DataMode(mode_str)
with col3:
    data_dir = st.text_input("Data directory", value=cfg.data_dir)

cfg.valuation_date = new_date
cfg.mode = new_mode
cfg.data_dir = data_dir

st.session_state["market_config"] = cfg

# --- Choix ticker equity ---
ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL")

# --- Equity data ---
st.subheader("📊 Equity (Spot & Historique)")

eq_conf = EquityConfig(ticker=ticker, history_years=2)
eq_mkt = EquityMarketData(cfg, eq_conf)

try:
    history = eq_mkt.history
    spot = eq_mkt.spot

    st.metric(label=f"Spot {ticker}", value=f"{spot:.2f} {cfg.currency}")

    st.line_chart(history["Adj Close"] if "Adj Close" in history.columns else history["Close"])
    st.caption("Historique des prix (Adj Close / Close)")

except Exception as e:
    st.error(f"Erreur lors du chargement des données equity : {e}")

# --- Options data ---
st.subheader("🧾 Option Chains")

max_mats = st.number_input("Nombre max de maturités à charger", min_value=1, max_value=20, value=5)

opt_conf = OptionChainConfig(ticker=ticker, max_maturities=max_mats)
opt_mkt = OptionChainMarketData(cfg, opt_conf)

try:
    maturities = opt_mkt.maturities
    st.write("Maturités disponibles :", maturities)

    mat_selected = st.selectbox("Choisir une maturité", options=maturities)
    calls, puts = opt_mkt.get_chain(mat_selected)

    tab1, tab2 = st.tabs(["Calls", "Puts"])
    with tab1:
        st.dataframe(calls.head(50))
    with tab2:
        st.dataframe(puts.head(50))

except Exception as e:
    st.error(f"Erreur lors du chargement des options : {e}")
