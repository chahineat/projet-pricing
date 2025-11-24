import numpy as np
from dataclasses import dataclass
from scipy.optimize import least_squares

from equity.black_scholes import BlackScholesModel
from equity.heston import HestonModel, HestonParams
from volatility.vol_surface import VolSurface


# ---------------------------
# Calibrer la volatilité BS
# ---------------------------
def calibrate_bs_iv(K, T, market_iv, spot, r):
    """Trouve sigma BS qui colle le mieux aux IV du marché."""
    K = np.array(K)
    iv = np.array(market_iv)

    def residuals(sig):
        bs = BlackScholesModel(spot, r, sig[0])
        model = np.array([bs.implied_vol(bs.call_price(k, T), k, T) for k in K])
        return model - iv

    res = least_squares(residuals, np.array([0.2]), bounds=(1e-4, 5.0))
    return res.x[0]


# ---------------------------
# Calibrer Heston par moindres carrés
# ---------------------------
def calibrate_heston(
    K: np.ndarray,
    T: float,
    market_iv: np.ndarray,
    spot: float,
    r: float,
    q: float = 0.0,
    initial: HestonParams = None,
):
    K = np.array(K)
    iv = np.array(market_iv)

    if initial is None:
        initial = HestonParams(kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5, v0=0.04)

    def residuals(x):
        params = HestonParams(kappa=x[0], theta=x[1], sigma=x[2], rho=np.tanh(x[3]), v0=x[4])
        model = HestonModel(spot, r, params, q)
        # Option: utiliser la formule semi-fermée plutôt que MC → à faire plus tard
        model_iv = np.array([_heston_iv(model, k, T) for k in K])
        return model_iv - iv

    x0 = np.array([initial.kappa, initial.theta, initial.sigma, np.arctanh(initial.rho), initial.v0])

    res = least_squares(residuals, x0, method="trf")
    sol = res.x
    return HestonParams(sol[0], sol[1], sol[2], np.tanh(sol[3]), sol[4])


def _heston_iv(model: HestonModel, K: float, T: float):
    """
    Approximations rapides :
      - on simule un petit nombre de paths
      - on en déduit une implied vol BS
    Améliorable plus tard avec la formule analytique de Heston.
    """
    price = model.price_call_mc(K, T, N_steps=100, N_paths=5000)
    bs = BlackScholesModel(model.S0, model.r, 0.2)
    iv = bs.implied_vol(price, K, T)
    return iv
