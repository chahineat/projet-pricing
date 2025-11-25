# pricer.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


class Pricer:

    @staticmethod
    def d1(option, sigma):
        params = option.params
        return (np.log(params.S0 / option.K) +
                (params.r + 0.5 * sigma**2) * params.T) / (sigma * np.sqrt(params.T))

    @staticmethod
    def d2(option, sigma):
        return Pricer.d1(option, sigma) - sigma * np.sqrt(option.params.T)

    # ---------------------------
    # Black-Scholes price
    # ---------------------------
    @staticmethod
    def bs_price(option, sigma):
        S0, r, T = option.params.S0, option.params.r, option.params.T
        K = option.K
        d1 = Pricer.d1(option, sigma)
        d2 = Pricer.d2(option, sigma)

        if option.option_type == "call":
            return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    # ---------------------------
    # Implied Vol
    # ---------------------------
    @staticmethod
    def implied_vol(option, market_price):
        def f(sigma):
            return Pricer.bs_price(option, sigma) - market_price

        try:
            return brentq(f, 1e-6, 5)   # solve f(sigma)=0
        except:
            return np.nan

    # ---------------------------
    # Greeks
    # ---------------------------
    @staticmethod
    def delta(option, sigma):
        d1 = Pricer.d1(option, sigma)
        return norm.cdf(d1) if option.option_type == "call" else norm.cdf(d1) - 1

    @staticmethod
    def gamma(option, sigma):
        d1 = Pricer.d1(option, sigma)
        S0, T = option.params.S0, option.params.T
        return norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

    @staticmethod
    def vega(option, sigma):
        d1 = Pricer.d1(option, sigma)
        T = option.params.T
        return option.params.S0 * norm.pdf(d1) * np.sqrt(T)

    @staticmethod
    def theta(option, sigma):
        S0, r, T = option.params.S0, option.params.r, option.params.T
        K = option.K
        d1 = Pricer.d1(option, sigma)
        d2 = Pricer.d2(option, sigma)

        theta_call = (
            -S0 * norm.pdf(d1) * sigma / (2*np.sqrt(T))
            - r * K * np.exp(-r*T) * norm.cdf(d2)
        )
        theta_put = (
            -S0 * norm.pdf(d1) * sigma / (2*np.sqrt(T))
            + r * K * np.exp(-r*T) * norm.cdf(-d2)
        )

        return theta_call if option.option_type == "call" else theta_put

    @staticmethod
    def rho(option, sigma):
        K, r, T = option.K, option.params.r, option.params.T
        d2 = Pricer.d2(option, sigma)
        rho_call = K * T * np.exp(-r*T) * norm.cdf(d2)
        rho_put = -K * T * np.exp(-r*T) * norm.cdf(-d2)
        return rho_call if option.option_type == "call" else rho_put
