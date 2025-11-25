import datetime as dt
import streamlit as st
import numpy as np

from market import MarketConfig, DataMode, EquityConfig, EquityMarketData
from equity.black_scholes import BlackScholesModel
from equity.heston import HestonParams, HestonModel
from equity.monte_carlo import monte_carlo_pricer, european_call_payoff, european_put_payoff


st.set_page_config(page_title="Pricing Tools", layout="wide")

st.title("🧮 Pricing Tools – Options")

# --- Config globale ---
if "market_config" not in st.session_state:
    st.session_state["market_config"] = MarketConfig(
        valuation_date=dt.date.today(),
        mode=DataMode.SNAPSHOT,
        currency="USD",
        data_dir="data",
    )

cfg: MarketConfig = st.session_state["market_config"]

ticker = st.text_input("Ticker", value="AAPL")
eq_mkt = EquityMarketData(cfg, EquityConfig(ticker=ticker))
S0 = eq_mkt.spot

col1, col2, col3 = st.columns(3)
with col1:
    K = st.number_input("Strike K", value=float(round(S0, 2)))
with col2:
    T = st.number_input("Maturité T (années)", value=0.5, min_value=0.01, max_value=10.0)
with col3:
    r = st.number_input("Taux sans risque r", value=0.03, min_value=-0.05, max_value=0.2)

tab_bs, tab_heston = st.tabs(["Black–Scholes", "Heston MC"])

with tab_bs:
    st.subheader("Black–Scholes analytique")
    sigma = st.number_input("Volatilité σ", value=0.30, min_value=0.0001, max_value=3.0)
    q = st.number_input("Dividende q", value=0.0, min_value=0.0, max_value=0.2)

    bs = BlackScholesModel(spot=S0, rate=r, volatility=sigma, dividend_yield=q)
    call_price = bs.call_price(K, T)
    put_price = bs.put_price(K, T)
    delta_call = bs.delta(K, T, option="call")
    delta_put = bs.delta(K, T, option="put")
    gamma = bs.gamma(K, T)
    vega = bs.vega(K, T)

    colp1, colp2 = st.columns(2)
    with colp1:
        st.metric("Call price", f"{call_price:.4f}")
        st.metric("Put price", f"{put_price:.4f}")
    with colp2:
        st.write(
            {
                "Delta (call)": delta_call,
                "Delta (put)": delta_put,
                "Gamma": gamma,
                "Vega": vega,
            }
        )

with tab_heston:
    st.subheader("Heston – Monte Carlo")

    colh1, colh2, colh3 = st.columns(3)
    with colh1:
        kappa = st.number_input("kappa", value=1.0)
        theta = st.number_input("theta", value=0.04)
    with colh2:
        sigma_v = st.number_input("sigma (vol of vol)", value=0.5)
        rho = st.number_input("rho", value=-0.5, min_value=-0.999, max_value=0.999)
    with colh3:
        v0 = st.number_input("v0 (variance initiale)", value=0.04)
        n_paths = st.number_input("N_paths", value=5000, min_value=1000, max_value=50000, step=1000)

    if st.button("Pricer Call & Put Heston MC"):
        params = HestonParams(kappa=kappa, theta=theta, sigma=sigma_v, rho=rho, v0=v0)
        heston = HestonModel(S0, r, params)
        S_paths, _ = heston.simulate_paths(T, N_steps=200, N_paths=int(n_paths))

        res_call = monte_carlo_pricer(S_paths, lambda S: european_call_payoff(S, K), r, T)
        res_put = monte_carlo_pricer(S_paths, lambda S: european_put_payoff(S, K), r, T)

        st.write("Call Heston MC:", res_call)
        st.write("Put Heston MC:", res_put)
