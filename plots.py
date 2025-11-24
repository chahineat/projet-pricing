# plots.py
import numpy as np
import matplotlib.pyplot as plt
from Market_data import MarketData
from equity_options_bs import BlackScholesModel


def plot_call_payoff(strike: float, S_min: float = 0, S_max: float = 500):
    S = np.linspace(S_min, S_max, 500)
    payoff = np.maximum(S - strike, 0.0)
    plt.figure()
    plt.plot(S, payoff)
    plt.xlabel("Prix du sous-jacent $S_T$")
    plt.ylabel("Payoff du call")
    plt.title(f"Payoff d'un call de strike K={strike}")
    plt.grid(True)
    plt.show()


def plot_put_payoff(strike: float, S_min: float = 0, S_max: float = 500):
    S = np.linspace(S_min, S_max, 500)
    payoff = np.maximum(strike - S, 0.0)
    plt.figure()
    plt.plot(S, payoff)
    plt.xlabel("Prix du sous-jacent $S_T$")
    plt.ylabel("Payoff du put")
    plt.title(f"Payoff d'un put de strike K={strike}")
    plt.grid(True)
    plt.show()


def plot_iv_smile(maturity_index: int = 3):
    """
    Trace un smile de volatilité implicite pour AAPL et une maturité donnée.
    """
    mkt = MarketData()
    maturities = mkt.get_available_maturities()
    maturity = maturities[maturity_index]

    calls, _ = mkt.get_option_chain(maturity)
    calls = calls[calls["impliedVolatility"] > 1e-4]

    plt.figure()
    plt.plot(calls["strike"], calls["impliedVolatility"])
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title(f"Smile de volatilité (AAPL, maturity={maturity})")
    plt.grid(True)
    plt.show()
