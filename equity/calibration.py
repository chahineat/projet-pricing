# equity/calibration.py
import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares

from equity.black_scholes import BlackScholesModel
from equity.heston import HestonModel, HestonParams


# ---------------------------
# Calibrer la volatilité BS
# ---------------------------

def calibrate_bs_iv(K, T, market_iv, spot, r):
    """
    Trouve une volatilité BS unique qui colle au mieux aux IV du marché
    (en moindres carrés).
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(market_iv, dtype=float)

    # Nettoyage basique
    mask = np.isfinite(K) & np.isfinite(iv) & (K > 0) & (iv > 0)
    K = K[mask]
    iv = iv[mask]

    if len(K) < 2:
        raise ValueError("Pas assez de points valides pour calibrer BS.")

    def residuals(sig):
        sigma = float(sig[0])
        if sigma <= 0:
            return 1e6 * np.ones_like(iv)
        bs = BlackScholesModel(spot, r, sigma)
        model_iv = []
        for k in K:
            # On passe par le prix BS puis on recalcule une IV pour être cohérent
            price = bs.call_price(k, T)
            iv_model_k = bs.implied_vol(price, k, T)
            if not np.isfinite(iv_model_k) or iv_model_k <= 0:
                iv_model_k = sigma  # fallback
            model_iv.append(iv_model_k)
        model_iv = np.array(model_iv)
        return model_iv - iv

    res = least_squares(residuals, np.array([0.2]), bounds=(1e-4, 5.0))
    return float(res.x[0])


# ---------------------------
# Heston : helper IV
# ---------------------------

def _heston_iv(model: HestonModel, K: float, T: float) -> float:
    """
    Approximation rapide d'une IV Heston :
      - price_call_mc
      - puis inversion via BS.
    Si ça plante, renvoie np.nan (géré par la calibration).
    """
    try:
        price = model.price_call_mc(K, T, N_steps=100, N_paths=5000)
    except Exception:
        return np.nan

    if price <= 0 or not np.isfinite(price):
        return np.nan

    # volatilité de base pour l'inversion
    bs = BlackScholesModel(model.S0, model.r, 0.2)
    iv = bs.implied_vol(price, K, T)
    if not np.isfinite(iv) or iv <= 0:
        return np.nan
    return iv


# ---------------------------
# Calibrer Heston
# ---------------------------

def calibrate_heston(
    K: np.ndarray,
    T: float,
    market_iv: np.ndarray,
    spot: float,
    r: float,
    q: float = 0.0,
    initial: HestonParams | None = None,
) -> HestonParams:
    """
    Calibration Heston (kappa, theta, sigma, rho, v0)
    par moindres carrés sur un smile (K, IV).

    ATTENTION: c'est une calibration approximative basée sur
    une inversion MC -> IV BS, ce n'est pas la formule fermée.
    """

    K = np.asarray(K, dtype=float)
    iv = np.asarray(market_iv, dtype=float)

    # Nettoyage de base : garder seulement les points valides
    mask = np.isfinite(K) & np.isfinite(iv) & (K > 0) & (iv > 0)
    K = K[mask]
    iv = iv[mask]

    if len(K) < 3:
        raise ValueError("Pas assez de points valides pour calibrer Heston (au moins 3).")

    if initial is None:
        initial = HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5, v0=0.04)

    def residuals(x):
        kappa, theta, sigma_v, rho_raw, v0 = x

        # Quelques contraintes soft:
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or v0 <= 0:
            return 1e6 * np.ones_like(iv)

        rho = np.tanh(rho_raw)  # force |rho|<1
        params = HestonParams(kappa=kappa, theta=theta, sigma=sigma_v, rho=rho, v0=v0)
        model = HestonModel(spot, r, params, q)

        model_iv = np.array([_heston_iv(model, k, T) for k in K])

        # Si des IV modèles sont NaN / inf → renvoyer un gros résidu mais FINI
        if not np.all(np.isfinite(model_iv)):
            return 1e6 * np.ones_like(iv)

        return model_iv - iv

    x0 = np.array(
        [
            initial.kappa,
            initial.theta,
            initial.sigma,
            np.arctanh(initial.rho),
            initial.v0,
        ]
    )

    # On met des bornes raisonnables pour éviter des paramètres dégénérés
    bounds_lower = [1e-4, 1e-6, 1e-4, -5.0, 1e-6]
    bounds_upper = [10.0, 2.0, 5.0, 5.0, 2.0]

    res = least_squares(residuals, x0, method="trf", bounds=(bounds_lower, bounds_upper))

    kappa_fit, theta_fit, sigma_fit, rho_raw_fit, v0_fit = res.x
    rho_fit = np.tanh(rho_raw_fit)

    return HestonParams(
        kappa=float(kappa_fit),
        theta=float(theta_fit),
        sigma=float(sigma_fit),
        rho=float(rho_fit),
        v0=float(v0_fit),
    )
