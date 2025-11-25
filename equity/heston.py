import numpy as np
from dataclasses import dataclass

@dataclass
class HestonParams:
    kappa: float     # vitesse de réversion
    theta: float     # variance long-terme
    sigma: float     # volatilité de la variance (vol of vol)
    rho: float       # corrélation W1/W2
    v0: float        # variance initiale


class HestonModel:
    def __init__(self, S0, r, params: HestonParams, q=0.0):
        self.S0 = S0
        self.r = r
        self.q = q
        self.params = params

    # ---------------------------
    # Simulation Heston (Euler)
    # ---------------------------
    def simulate_paths(self, T, N_steps=252, N_paths=20000, seed=0):
        np.random.seed(seed)
        dt = T / N_steps

        kappa, theta, sigma, rho, v0 = (
            self.params.kappa,
            self.params.theta,
            self.params.sigma,
            self.params.rho,
            self.params.v0,
        )

        S = np.zeros((N_paths, N_steps + 1))
        v = np.zeros((N_paths, N_steps + 1))
        S[:, 0] = self.S0
        v[:, 0] = v0

        for t in range(1, N_steps + 1):
            Z1 = np.random.normal(size=N_paths)
            Z2 = np.random.normal(size=N_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

            # variance: CIR discretisation
            v_prev = v[:, t - 1]
            v_new = v_prev + kappa * (theta - v_prev) * dt + sigma * np.sqrt(np.maximum(v_prev, 0)) * np.sqrt(dt) * Z2
            v_new = np.maximum(v_new, 1e-8)
            v[:, t] = v_new

            # prix action
            S[:, t] = S[:, t - 1] * np.exp((self.r - self.q - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * Z1)

        return S, v

    # ---------------------------
    # Pricing Call MC
    # ---------------------------
    def price_call_mc(self, K, T, N_steps=252, N_paths=20000):
        S, _ = self.simulate_paths(T, N_steps, N_paths)
        payoffs = np.maximum(S[:, -1] - K, 0)
        return np.exp(-self.r * T) * payoffs.mean()
