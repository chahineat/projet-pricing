from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    params: SABRParams,
    epsilon: float = 1e-07,
) -> float:
    """
    Formule approchée de Hagan pour la vol implicite SABR (beta-modèle).

    F : forward
    K : strike
    T : maturité (en années)
    params : SABRParams(alpha, beta, rho, nu)
    """
    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu

    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan

    if abs(F - K) < epsilon:
        # Cas ATM
        FK_beta = F ** (1 - beta)
        term1 = alpha / (FK_beta)
        term2 = (
            ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
            + (rho * beta * nu * alpha / (4 * F ** (1 - beta)))
            + ((2 - 3 * rho ** 2) * nu ** 2 / 24)
        ) * T
        return term1 * (1 + term2)

    # Cas K != F
    logFK = np.log(F / K)
    FK_beta = (F * K) ** ((1 - beta) / 2)

    z = (nu / alpha) * FK_beta * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    num = alpha * z
    den = FK_beta * x_z

    # Correctif en T
    term1 = ((1 - beta) ** 2 / 24) * (logFK ** 2)
    term2 = ((1 - beta) ** 4 / 1920) * (logFK ** 4)
    A = 1 + (term1 + term2)

    B = (
        ((1 - beta) ** 2 / 24) * (alpha ** 2 / (F ** (2 - 2 * beta)))
        + (rho * beta * nu * alpha / (4 * F ** (1 - beta)))
        + ((2 - 3 * rho ** 2) * nu ** 2 / 24)
    ) * T

    return (num / den) * A * (1 + B)


def calibrate_sabr_to_smile(
    K: np.ndarray,
    iv: np.ndarray,
    F: float,
    T: float,
    beta: float = 0.5,
    initial_alpha: float = 0.2,
    initial_rho: float = 0.0,
    initial_nu: float = 0.5,
) -> SABRParams:
    """
    Calibre SABR (alpha, rho, nu) pour un beta donné à un smile (K, iv).
    Utilise least_squares de SciPy.

    K : strikes
    iv : volatilities observées (décimal)
    F : forward
    T : maturité (année)
    """
    K = np.asarray(K, dtype=float)
    iv = np.asarray(iv, dtype=float)

    def residuals(x):
        a, r, n = x
        params = SABRParams(alpha=max(a, 1e-6), beta=beta, rho=np.tanh(r), nu=max(n, 1e-6))
        model_iv = np.array([sabr_implied_vol(F, k, T, params) for k in K])
        return model_iv - iv

    x0 = np.array([initial_alpha, initial_rho, initial_nu])
    res = least_squares(residuals, x0, method="trf")

    a_fit, r_fit, n_fit = res.x
    params_fit = SABRParams(alpha=max(a_fit, 1e-6), beta=beta, rho=np.tanh(r_fit), nu=max(n_fit, 1e-6))
    return params_fit
