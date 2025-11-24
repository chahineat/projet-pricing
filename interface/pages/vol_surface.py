import datetime as dt
import streamlit as st

from market import MarketConfig, DataMode, EquityConfig
from volatility.extract_surface import SurfaceExtractionConfig, extract_vol_surface
from volatility.vol_surface import VolSurface
from volatility.vol_smile import smile_from_surface
from volatility.plots.smile_plots import plot_smile
from volatility.plots.surface_plots import plot_vol_surface


st.set_page_config(page_title="Volatility Surface", layout="wide")

st.title("🌀 Volatility – Smiles & Surface")

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

col1, col2 = st.columns(2)
with col1:
    max_mats = st.number_input("Max maturities", min_value=1, max_value=20, value=5)
with col2:
    min_iv = st.number_input("Min IV filter", min_value=0.0, max_value=1.0, value=0.0001, step=0.0001)

extract_conf = SurfaceExtractionConfig(
    ticker=ticker,
    max_maturities=max_mats,
    min_iv=min_iv,
    use_calls=True,
)

if st.button("Extraire surface"):
    try:
        surf_df = extract_vol_surface(cfg, extract_conf, eq_conf=EquityConfig(ticker=ticker))
        st.success(f"Surface extraite ({len(surf_df)} points).")
        st.dataframe(surf_df.head())

        surface = VolSurface(surf_df)

        st.subheader("📈 Smile pour une maturité choisie")
        T_choice = st.selectbox(
            "Choisir maturité (T en années proche)",
            options=sorted(surface.maturities),
            index=0,
        )
        smile = smile_from_surface(surface, T_choice)

        with st.expander("Données du smile"):
            st.dataframe(smile.sorted())

        fig_smile = plot_smile(smile, title_prefix="Volatility smile")
        st.pyplot(fig_smile)

        st.subheader("🌈 Surface 3D interpolée")
        fig_surf = plot_vol_surface(surface)
        st.pyplot(fig_surf)

        st.session_state["vol_surface"] = surface

    except Exception as e:
        st.error(f"Erreur lors de l'extraction/affichage de la surface : {e}")
else:
    st.info("Clique sur 'Extraire surface' pour construire et visualiser la surface de volatilité.")
