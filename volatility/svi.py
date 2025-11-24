from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """
    Formule SVI classique:
        w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )
    où w(k) = sigma_imp(k)^2 * T (variance totale).
    """
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
    k_m = k - m
    return a + b * (rho * k_m + np.sqrt(k_m ** 2 + sigma ** 2))


def svi_implied_vol(k: np.ndarray, T: float, params: SVIParams) -> np.ndarray:
    """
    Vol implicite depuis la variance totale SVI:
        sigma_imp(k) = sqrt( w(k) / T )
    """
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 0.0) / T)


def calibrate_svi_to_smile(
    K: np.ndarray,
    iv: np.ndarray,
    F: float,
    T: float,
    initial_params: Tuple[float, float, float, float, float] = (0.01, 0.1, 0.0, 0.0, 0.1),
) -> SVIParams:
    """
    Calibre SVI sur un smile (K, iv) à maturité T donné.
    On travaille en log-moneyness k = ln(K/F).
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)
    k = np.log(K / F)

    def residuals(x):
        a, b, rho, m, sigma = x
        # petites contraintes soft: b,sigma>0, |rho|<1 via reparam si tu veux plus tard
        params = SVIParams(a=a, b=max(b, 1e-6), rho=np.tanh(rho), m=m, sigma=max(sigma, 1e-6))
        model_iv = svi_implied_vol(k, T, params)
        return model_iv - iv

    res = least_squares(residuals, np.array(initial_params), method="trf")
    a_fit, b_fit, rho_fit, m_fit, sigma_fit = res.x
    params_fit = SVIParams(
        a=a_fit,
        b=max(b_fit, 1e-6),
        rho=np.tanh(rho_fit),
        m=m_fit,
        sigma=max(sigma_fit, 1e-6),
    )
    return params_fit
