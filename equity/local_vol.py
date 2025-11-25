import numpy as np
from dataclasses import dataclass
from volatility.vol_surface import VolSurface
from equity.black_scholes import BlackScholesModel


@dataclass
class LocalVolSurface:
    Ks: np.ndarray
    Ts: np.ndarray
    sigmas: np.ndarray    # matrice (len(Ts) × len(Ks))

def compute_local_vol_surface(surface: VolSurface, S0: float, r: float, q: float = 0.0,
                              n_T=20, n_K=40):
    """
    Approxime la volatilité locale via Dupire sur une grille (T,K).
    Utilise Black–Scholes pour transformer vol implicite -> prix call.
    """
    Ts = np.linspace(min(surface.maturities), max(surface.maturities), n_T)
    Ks = np.linspace(surface.raw["K"].min(), surface.raw["K"].max(), n_K)

    lv = np.zeros((n_T, n_K))

    for i, T in enumerate(Ts):
        for j, K in enumerate(Ks):
            # prix call via BS avec IV interpolé
            iv = surface.iv_at(K, T)
            bs = BlackScholesModel(S0, r, iv, q)
            C = bs.call_price(K, T)

            # approx dC/dT et d2C/dK2 par FD centrée
            epsT = 1e-3
            epsK = K * 0.01

            iv_T_plus = surface.iv_at(K, T + epsT)
            iv_T_minus = surface.iv_at(K, T - epsT)
            C_T_plus = BlackScholesModel(S0, r, iv_T_plus, q).call_price(K, T + epsT)
            C_T_minus = BlackScholesModel(S0, r, iv_T_minus, q).call_price(K, T - epsT)
            dC_dT = (C_T_plus - C_T_minus) / (2 * epsT)

            C_K_plus = bs.call_price(K + epsK, T)
            C_K_minus = bs.call_price(K - epsK, T)
            d2C_dK2 = (C_K_plus - 2 * C + C_K_minus) / (epsK**2)

            denom = 0.5 * K * K * d2C_dK2
            if denom <= 0:
                lv[i, j] = np.nan
            else:
                lv[i, j] = np.sqrt(np.maximum(dC_dT / denom, 0))

    return LocalVolSurface(Ks=Ks, Ts=Ts, sigmas=lv)
