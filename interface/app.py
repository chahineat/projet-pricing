import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import erf
from equity import heston_analytic

@dataclass
class HestonParams:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

def heston_char_func(
    u: complex,
    T: float,
    S0: float,
    r: float,
    q: float,
    params: HestonParams,
) -> complex:
    kappa = params.kappa
    theta = params.theta
    sigma = params.sigma
    rho = params.rho
    v0 = params.v0

    x0 = np.log(S0)

    iu = 1j * u
    a = kappa * theta
    b = kappa

    d = np.sqrt((rho * sigma * iu - b) ** 2 + (sigma ** 2) * (iu + u ** 2))
    g = (b - rho * sigma * iu + d) / (b - rho * sigma * iu - d)

    exp_dT = np.exp(d * T)
    one_minus_gexp = 1.0 - g * exp_dT
    one_minus_g = 1.0 - g

    C = (
        iu * (x0 + (r - q) * T)
        + (a / (sigma ** 2)) * (
            (b - rho * sigma * iu + d) * T
            - 2.0 * np.log(one_minus_gexp / one_minus_g)
        )
    )
    D = ((b - rho * sigma * iu + d) / (sigma ** 2)) * ((1.0 - exp_dT) / one_minus_gexp)

    return np.exp(C + D * v0)


def _heston_Pj(
    j: int,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    u_max: float = 100.0,
) -> float:
    if j not in (1, 2):
        raise ValueError("j must be 1 or 2")

    logK = np.log(K)

    def integrand(u: float) -> float:
        u_complex = u - 1e-14j
        if j == 1:
            phi = heston_char_func(u_complex - 1j, T, S0, r, q, params)
        else:
            phi = heston_char_func(u_complex, T, S0, r, q, params)

        num = np.exp(-1j * u_complex * logK) * phi
        denom = 1j * u_complex
        return np.real(num / denom)

    from scipy.integrate import quad
    integral, _ = quad(integrand, 0.0, u_max, limit=200)
    return 0.5 + (1.0 / np.pi) * integral


def heston_call_price_cf(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    u_max: float = 100.0,
) -> float:
    if T <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    P1 = _heston_Pj(1, S0, K, T, r, q, params, u_max=u_max)
    P2 = _heston_Pj(2, S0, K, T, r, q, params, u_max=u_max)

    return float(S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2)



st.set_page_config(page_title="Option Pricing Lab", layout="wide")
st.title("📍 Option Pricing Home — Price, Greeks & Payoff")


# -------------------------------------------------------------------
# Black–Scholes helpers (self-contained)
# -------------------------------------------------------------------
def norm_cdf(x: np.ndarray) -> np.ndarray:
    # simple approximation via erf (no scipy dependency)
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def bs_price(S0, K, T, r, q, sigma, option_type: str):
    if T <= 0:
        intrinsic = max(0.0, (S0 - K) if option_type == "Call" else (K - S0))
        return intrinsic

    if sigma <= 0:
        # degenerate case: almost deterministic
        forward = S0 * np.exp((r - q) * T)
        disc = np.exp(-r * T)
        ST = forward  # crude
        intrinsic = max(0.0, (ST - K) if option_type == "Call" else (K - ST))
        return disc * intrinsic

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = (S0 * np.exp(-q * T) * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm_cdf(-d2) - S0 * np.exp(-q * T) * norm_cdf(-d1))

    return float(price)


def bs_greeks(S0, K, T, r, q, sigma, option_type: str):
    if T <= 0 or sigma <= 0:
        # fallback numerical approx
        return numerical_greeks(lambda s, v: bs_price(s, K, T, r, q, v, option_type), S0, sigma)

    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 ** 2)

    if option_type == "Call":
        delta = np.exp(-q * T) * norm_cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm_cdf(d1) - 1.0)

    gamma = (np.exp(-q * T) * pdf_d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
    theta = (
        - (S0 * np.exp(-q * T) * pdf_d1 * sigma) / (2.0 * np.sqrt(T))
        - (r * K * np.exp(-r * T) * (norm_cdf(d2) if option_type == "Call" else norm_cdf(-d2)))
        + (q * S0 * np.exp(-q * T) * (norm_cdf(d1) if option_type == "Call" else norm_cdf(-d1)))
    )
    rho = (
        K * T * np.exp(-r * T) * (norm_cdf(d2) if option_type == "Call" else -norm_cdf(-d2))
    )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


# -------------------------------------------------------------------
# Numerical Greeks helper (for Heston or fallback)
# -------------------------------------------------------------------
def numerical_greeks(price_fn, S0, vol_or_v0, eps_s=1e-3, eps_v=1e-3):
    # delta & gamma wrt S
    p_plus = price_fn(S0 + eps_s, vol_or_v0)
    p_minus = price_fn(S0 - eps_s, vol_or_v0)
    p0 = price_fn(S0, vol_or_v0)

    delta = (p_plus - p_minus) / (2.0 * eps_s)
    gamma = (p_plus - 2.0 * p0 + p_minus) / (eps_s ** 2)

    # vega wrt vol_or_v0 (for BS = sigma, for Heston ~ v0)
    pv_plus = price_fn(S0, vol_or_v0 + eps_v)
    pv_minus = price_fn(S0, vol_or_v0 - eps_v)
    vega = (pv_plus - pv_minus) / (2.0 * eps_v)

    # theta, rho not handled here → set to NaN or 0
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float("nan"),
        "rho": float("nan"),
    }


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
st.markdown(
    """
Cette page est le **point d'entrée** : choisis le type d'option, le modèle,
paramètre tout, et obtiens **prix, greeks et payoff**.  
Les pages suivantes serviront pour la **calibration** et la **surface de volatilité**.
"""
)

col_model, col_type = st.columns(2)
with col_model:
    model_choice = st.selectbox("Modèle", ["Black–Scholes", "Heston (CF)"])
with col_type:
    option_type = st.selectbox("Type d'option", ["Call", "Put"])

st.markdown("### Paramètres du contrat et du marché")

col1, col2, col3 = st.columns(3)
with col1:
    S0 = st.number_input("Spot S₀", value=100.0, min_value=0.01)
    K = st.number_input("Strike K", value=100.0, min_value=0.01)
with col2:
    T = st.number_input("Maturité T (années)", value=1.0, min_value=0.0001)
    r = st.number_input("Taux sans risque r (continu)", value=0.01, step=0.001)
with col3:
    q = st.number_input("Dividende continu q", value=0.0, step=0.001)

st.markdown("### Paramètres du modèle")

if model_choice == "Black–Scholes":
    sigma = st.number_input("Volatilité σ", value=0.2, min_value=0.0001)
    params_heston = None
else:
    # Heston params
    colh1, colh2 = st.columns(2)
    with colh1:
        kappa = st.number_input("kappa (mean reversion)", value=1.0)
        theta = st.number_input("theta (variance long terme)", value=0.04)
        sigma_v = st.number_input("sigma_v (vol of vol)", value=0.5)
    with colh2:
        rho = st.number_input("rho (corrélation)", value=-0.5, min_value=-0.99, max_value=0.99, step=0.01)
        v0 = st.number_input("v0 (variance initiale)", value=0.04)

    params_heston = HestonParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma_v),
        rho=float(rho),
        v0=float(v0),
    )
    sigma = None  # not used


if st.button("💡 Calculer prix, greeks et payoff"):
    try:
        # ------------------------------------------------------------
        # Prix
        # ------------------------------------------------------------
        if model_choice == "Black–Scholes":
            price = bs_price(S0, K, T, r, q, sigma, option_type)
        else:
            # Heston call via CF + put-call parity if needed
            call_price = heston_call_price_cf(
                S0=S0, K=K, T=T, r=r, q=q, params=params_heston
            )
            if option_type == "Call":
                price = call_price
            else:
                # Put-Call parity: P = C - S e^{-qT} + K e^{-rT}
                price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)

        # ------------------------------------------------------------
        # Greeks (BS analytique, Heston numérique)
        # ------------------------------------------------------------
        if model_choice == "Black–Scholes":
            greeks = bs_greeks(S0, K, T, r, q, sigma, option_type)
        else:
            def price_given_S_and_v0(S, v_init):
                # Use same Heston params but with modified S or v0
                params_tmp = HestonParams(
                    kappa=params_heston.kappa,
                    theta=params_heston.theta,
                    sigma=params_heston.sigma,
                    rho=params_heston.rho,
                    v0=v_init,
                )
                call_p = heston_call_price_cf(S0=S, K=K, T=T, r=r, q=q, params=params_tmp)
                if option_type == "Call":
                    return call_p
                else:
                    return call_p - S * np.exp(-q * T) + K * np.exp(-r * T)

            greeks = numerical_greeks(price_given_S_and_v0, S0, params_heston.v0)

        # ------------------------------------------------------------
        # Affichage des résultats
        # ------------------------------------------------------------
        st.subheader("📊 Résultats")

        colA, colB, colC = st.columns(3)
        colA.metric("Prix", f"{price:.4f}")
        colB.metric("Delta", f"{greeks['delta']:.4f}")
        colC.metric("Gamma", f"{greeks['gamma']:.4f}")

        colD, colE = st.columns(2)
        colD.metric("Vega", f"{greeks['vega']:.4f}")
        colE.metric("Theta", f"{greeks['theta']:.4f}" if not np.isnan(greeks['theta']) else "N/A")

        # ------------------------------------------------------------
        # Payoff plot
        # ------------------------------------------------------------
        st.subheader("💰 Payoff à maturité")

        S_T = np.linspace(0.0, 2.0 * S0, 200)
        if option_type == "Call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)

        fig, ax = plt.subplots()
        ax.plot(S_T, payoff, label=f"Payoff {option_type}")
        ax.axvline(K, linestyle="--", alpha=0.5, label="Strike")
        ax.set_xlabel("Sous-jacent à maturité S_T")
        ax.set_ylabel("Payoff")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors du calcul : {e}")
else:
    st.info("Renseigne les paramètres puis clique sur le bouton.")
