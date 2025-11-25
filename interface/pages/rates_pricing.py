import datetime as dt
import streamlit as st
import pandas as pd

from market import MarketConfig, DataMode, RatesMarketData, RatesConfig
from rates.bootstrap_curve import bootstrap_from_zero_rates
from rates.bond_pricing import CouponBond
from rates.swap_pricing import InterestRateSwap
from rates.futures import equity_future_price

st.set_page_config(page_title="Rates Pricing", layout="wide")
st.title("🏦 Rates – Pricing des instruments de taux")

# Config marché partagée
if "market_config" not in st.session_state:
    st.session_state["market_config"] = MarketConfig(
        valuation_date=dt.date.today(),
        mode=DataMode.LIVE,
        currency="USD",
        data_dir="data",
    )

cfg: MarketConfig = st.session_state["market_config"]

curve_name = st.text_input("Nom de la courbe", value="USD_ZERO")
rates_mkt = RatesMarketData(cfg, RatesConfig(curve_name=curve_name))
raw_df = rates_mkt.raw_curve
curve = bootstrap_from_zero_rates(raw_df, col_maturity="maturity", col_rate="rate", rate_is_continuous=True)

st.subheader("📄 Courbe utilisée")
st.dataframe(raw_df)

# ---- Bond ----
st.subheader("📌 Bond à coupons")

nominal = st.number_input("Nominal", value=100.0)
coupon = st.number_input("Coupon annuel (%)", value=4.0) / 100.0
mat_bond = st.number_input("Maturité bond (années)", value=5.0)
freq = st.number_input("Fréquence (nb coupons/an)", value=1, min_value=1, max_value=4, step=1)

bond = CouponBond(nominal=nominal, coupon_rate=coupon, maturity=mat_bond, frequency=freq)
st.write(f"Prix du bond : {bond.price(curve):.4f}")

# ---- Swap ----
st.subheader("📌 Swap de taux (payer fixe)")

notional = st.number_input("Notional swap", value=100.0)
fixed_rate = st.number_input("Taux fixe (%)", value=4.0) / 100.0
swap_mat = st.number_input("Maturité swap (années)", value=5.0)
year_frac = st.number_input("Year fraction", value=1.0)

payment_dates = [year_frac * i for i in range(1, int(swap_mat / year_frac) + 1)]
swap = InterestRateSwap(
    notional=notional,
    fixed_rate=fixed_rate,
    payment_times=payment_dates,
    year_fraction=year_frac,
)
st.write(f"NPV du payer swap : {swap.npv_payer(curve):.4f}")

# ---- Future equity ----
st.subheader("📌 Future sur action (pricing simple)")

S0 = st.number_input("Spot equity", value=170.0)
r = st.number_input("Taux sans risque r", value=0.03)
T_fut = st.number_input("Maturité future (années)", value=1.0)
st.write(f"Prix théorique future : {equity_future_price(S0, r, T_fut):.4f}")
