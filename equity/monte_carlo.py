import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class MonteCarloResult:
    price: float
    stderr: float
    conf_int: tuple


def monte_carlo_pricer(
    S_paths: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    r: float,
    T: float,
):
    """
    S_paths: matrice (N_paths × N_steps+1)
    payoff_fn: prend un vecteur de prix finaux, ou la trajectoire complète
    """
    payoffs = payoff_fn(S_paths)
    disc = np.exp(-r * T)
    price = disc * payoffs.mean()

    stderr = disc * payoffs.std(ddof=1) / np.sqrt(len(payoffs))
    ci = (price - 1.96 * stderr, price + 1.96 * stderr)

    return MonteCarloResult(price=price, stderr=stderr, conf_int=ci)


def european_call_payoff(S_paths: np.ndarray, K: float):
    return np.maximum(S_paths[:, -1] - K, 0.0)


def european_put_payoff(S_paths: np.ndarray, K: float):
    return np.maximum(K - S_paths[:, -1], 0.0)
