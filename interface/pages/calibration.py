import datetime as dt
import numpy as np
import streamlit as st

from market import MarketConfig, DataMode, EquityConfig, EquityMarketData
from volatility.vol_surface import VolSurface
from volatility.vol_smile import smile_from_surface
from volatility.sabr import calibrate_sabr_to_smile, sabr_implied_vol
from volatility.svi import calibrate_svi_to_smile, svi_implied_vol
from volatility.plots.smile_plots import plot_smile
from equity.black_scholes import BlackScholesModel
from equity.heston import HestonParams, HestonModel
from equity.calibration import calibrate_heston


st.set_page_config(page_title="Calibration", layout="wide")

st.title("🎯 Calibration des modèles")

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

# Il faut une surface déjà extraite (depuis la page 'Volatility Surface')
surface: VolSurface = st.session_state.get("vol_surface", None)
if surface is None:
    st.warning(
        "Aucune surface dans la session. Va d'abord dans la page 'Volatility Surface' "
        "et clique sur 'Extraire surface'."
    )
    st.stop()

eq_mkt = EquityMarketData(cfg, EquityConfig(ticker=ticker))
S0 = eq_mkt.spot
r = 0.0  # tu peux remplacer par un vrai r (courbe de taux)

T_choice = st.selectbox("Choisir une maturité pour la calibration", options=sorted(surface.maturities))
smile = smile_from_surface(surface, T_choice)
data = smile.sorted()
K = data["K"].to_numpy()
iv_mkt = data["iv"].to_numpy()

st.subheader("📈 Smile de marché")
st.pyplot(plot_smile(smile, title_prefix="Smile de marché"))

tab_bs, tab_sabr, tab_svi, tab_heston = st.tabs(["BS", "SABR", "SVI", "Heston"])

with tab_bs:
    st.markdown("### Calibration simple Black–Scholes (vol unique)")
    from equity.calibration import calibrate_bs_iv

    sigma_bs = calibrate_bs_iv(K, T_choice, iv_mkt, S0, r)
    st.write(f"Vol BS calibrée (unique) : {sigma_bs:.4f}")

with tab_sabr:
    st.markdown("### Calibration SABR sur le smile")
    from volatility.sabr import SABRParams

    beta = st.slider("β (beta)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    if st.button("Calibrer SABR"):
        params = calibrate_sabr_to_smile(K, iv_mkt, F=S0, T=T_choice, beta=beta)
        st.write("Paramètres SABR calibrés :", params)

        iv_model = np.array([sabr_implied_vol(S0, k, T_choice, params) for k in K])
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(K, iv_mkt, "o", label="Marché")
        ax.plot(K, iv_model, "-", label="SABR fit")
        ax.set_xlabel("K")
        ax.set_ylabel("IV")
        ax.legend()
        st.pyplot(fig)

with tab_svi:
    st.markdown("### Calibration SVI sur le smile")
    if st.button("Calibrer SVI"):
        from volatility.svi import calibrate_svi_to_smile, SVIParams, svi_implied_vol

        params_svi = calibrate_svi_to_smile(K, iv_mkt, F=S0, T=T_choice)
        st.write("Paramètres SVI calibrés :", params_svi)

        k_log = np.log(K / S0)
        iv_model_svi = svi_implied_vol(k_log, T_choice, params_svi)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(K, iv_mkt, "o", label="Marché")
        ax.plot(K, iv_model_svi, "-", label="SVI fit")
        ax.set_xlabel("K")
        ax.set_ylabel("IV")
        ax.legend()
        st.pyplot(fig)

with tab_heston:
    st.markdown("### Calibration Heston (simple, via MC)")

    kappa = st.number_input("kappa", value=1.0)
    theta = st.number_input("theta", value=0.04)
    sigma_v = st.number_input("sigma (vol of vol)", value=0.5)
    rho = st.number_input("rho", value=-0.5)
    v0 = st.number_input("v0 (variance initiale)", value=0.04)

    if st.button("Calibrer Heston (approximatif)"):
        initial = HestonParams(kappa=kappa, theta=theta, sigma=sigma_v, rho=rho, v0=v0)
        params_heston = calibrate_heston(K, T_choice, iv_mkt, S0, r, initial=initial)
        st.write("Paramètres Heston calibrés :", params_heston)
