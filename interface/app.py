import datetime as dt
import streamlit as st

from market import MarketConfig, DataMode

st.set_page_config(
    page_title="Pricing Project Dashboard",
    layout="wide",
)

def get_default_config() -> MarketConfig:
    """Config par défaut pour l'app (modifiable dans les pages)."""
    today = dt.date.today()
    return MarketConfig(
        valuation_date=today,
        mode=DataMode.LIVE,
        currency="USD",
        data_dir="data",
    )

st.session_state.setdefault("market_config", get_default_config())

st.title("📊 Pricing Project Dashboard")
st.markdown(
    """
Bienvenue dans l’interface de ton projet de **pricing taux & equity**.

Utilise le menu à gauche pour naviguer entre :

- **Market Data** : données actions & options (AAPL par exemple)
- **Rates Viewer** : courbe de taux, discount factors, forwards
- **Volatility Surface** : smiles & surface de volatilité implicite
- **Calibration** : calibration BS / SABR / SVI / Heston
- **Pricing Tools** : pricing d’options (BS, MC Heston, etc.)

La config marché courante (date, mode SNAPSHOT/LIVE, data_dir) est stockée
dans `st.session_state["market_config"]` et partagée entre les pages.
"""
)

st.subheader("⚙️ Configuration globale (résumé)")

cfg = st.session_state["market_config"]
st.write(
    {
        "valuation_date": cfg.valuation_date.isoformat(),
        "mode": cfg.mode.value,
        "currency": cfg.currency,
        "data_dir": cfg.data_dir,
    }
)

st.info(
    "Tu peux gérer la configuration détaillée dans chaque page (par exemple changer la date, le mode SNAPSHOT/LIVE, etc.)."
)
