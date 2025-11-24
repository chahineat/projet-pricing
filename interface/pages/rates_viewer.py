import datetime as dt
import streamlit as st
import pandas as pd

from market import MarketConfig, DataMode, RatesMarketData, RatesConfig
from rates.bootstrap_curve import bootstrap_from_zero_rates
from rates.discount_factors import DiscountCurve


st.set_page_config(page_title="Rates Viewer", layout="wide")

st.title("🏦 Rates Viewer – Courbe de taux & Discount Factors")

# --- Config globale ---
if "market_config" not in st.session_state:
    st.session_state["market_config"] = MarketConfig(
        valuation_date=dt.date.today(),
        mode=DataMode.SNAPSHOT,
        currency="USD",
        data_dir="data",
    )

cfg: MarketConfig = st.session_state["market_config"]

col1, col2 = st.columns(2)
with col1:
    curve_name = st.text_input("Nom de la courbe (ex: USD_ZERO)", value="USD_ZERO")
with col2:
    st.write(f"Data dir: `{cfg.data_dir}` – valuation_date: {cfg.valuation_date.isoformat()}")

# --- Chargement des taux bruts ---
rates_mkt = RatesMarketData(cfg, RatesConfig(curve_name=curve_name))

try:
    raw_df = rates_mkt.raw_curve
    st.subheader("📄 Données brutes de courbe (snapshot CSV)")
    st.dataframe(raw_df)

    curve = bootstrap_from_zero_rates(
        raw_df,
        col_maturity="maturity",
        col_rate="rate",
        rate_is_continuous=True,  # adapte selon tes données
    )

    st.subheader("📉 Zero-coupon curve (taux continus)")
    zero_points = pd.DataFrame(
        {
            "T": curve.maturities,
            "zero_rate": [curve.zero_rate(T) for T in curve.maturities],
        }
    )
    st.line_chart(zero_points.set_index("T"))

    st.subheader("📉 Discount factors")
    df_points = pd.DataFrame(
        {
            "T": curve.maturities,
            "DF": [curve.df(T) for T in curve.maturities],
        }
    )
    st.line_chart(df_points.set_index("T"))

except Exception as e:
    st.error(f"Erreur lors du chargement/bootstrapping de la courbe : {e}")
    st.info(
        "Vérifie que le fichier CSV existe dans data/rates et contient au moins "
        "les colonnes 'maturity' et 'rate'."
    )
