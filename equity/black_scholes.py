import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

class BlackScholesModel:
    """
    Modèle Black–Scholes classique en taux continus.
    SDE:
        dS_t = S_t * (r - q) dt + S_t * sigma dW_t
    """

    def __init__(self, spot: float, rate: float, volatility: float, dividend_yield: float = 0.0):
        self.S0 = spot
        self.r = rate
        self.sigma = volatility
        self.q = dividend_yield

    # ---------------------------
    #     Prix Black–Scholes
    # ---------------------------
    def d1(self, K, T):
        return (math.log(self.S0 / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (self.sigma * math.sqrt(T))

    def d2(self, K, T):
        return self.d1(K, T) - self.sigma * math.sqrt(T)

    def call_price(self, K, T):
        d1, d2 = self.d1(K, T), self.d2(K, T)
        return self.S0 * math.exp(-self.q * T) * norm.cdf(d1) - K * math.exp(-self.r * T) * norm.cdf(d2)

    def put_price(self, K, T):
        d1, d2 = self.d1(K, T), self.d2(K, T)
        return K * math.exp(-self.r * T) * norm.cdf(-d2) - self.S0 * math.exp(-self.q * T) * norm.cdf(-d1)

    # ---------------------------
    #         Greeks
    # ---------------------------
    def delta(self, K, T, option="call"):
        d1 = self.d1(K, T)
        if option == "call":
            return math.exp(-self.q * T) * norm.cdf(d1)
        else:
            return math.exp(-self.q * T) * (norm.cdf(d1) - 1)

    def gamma(self, K, T):
        d1 = self.d1(K, T)
        return math.exp(-self.q * T) * norm.pdf(d1) / (self.S0 * self.sigma * math.sqrt(T))

    def vega(self, K, T):
        d1 = self.d1(K, T)
        return self.S0 * math.exp(-self.q * T) * norm.pdf(d1) * math.sqrt(T)

    # ---------------------------
    #   Implied volatility
    # ---------------------------
    def implied_vol(self, market_price, K, T):
        def f(sig):
            self.sigma = sig
            return self.call_price(K, T) - market_price

        try:
            return brentq(f, 1e-6, 3.0)
        except:
            return float("nan")

    # ---------------------------
    #  Simulation de trajectoires
    # ---------------------------
    def simulate_paths(self, T, N_steps=252, N_paths=10000, seed=42):
        np.random.seed(seed)
        dt = T / N_steps
        increments = np.random.normal(size=(N_paths, N_steps))
        S = np.zeros((N_paths, N_steps + 1))
        S[:, 0] = self.S0

        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * np.sqrt(dt)

        for t in range(1, N_steps + 1):
            Z = increments[:, t - 1]
            S[:, t] = S[:, t - 1] * np.exp(drift + vol * Z)

        return S
