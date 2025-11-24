import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# -----------------------------
# Classe Parameters
# -----------------------------


class Parameters:
    def __init__(self, S0, r, T):
        self.S0 = S0
        self.r = r
        self.T = T

# -----------------------------
# Classe Option (abstraite)
# -----------------------------


class Option:
    def __init__(self, K, parameters):
        self.K = K
        self.params = parameters

    def black_scholes_price(self, sigma):
        d1 = (np.log(self.params.S0 / self.K) +
              (self.params.r + 0.5*sigma**2)*self.params.T) / (sigma*np.sqrt(self.params.T))
        d2 = d1 - sigma*np.sqrt(self.params.T)
        if self.option_type == "call":
            return self.params.S0*norm.cdf(d1) - self.K*np.exp(-self.params.r*self.params.T)*norm.cdf(d2)
        elif self.option_type == "put":
            return self.K*np.exp(-self.params.r*self.params.T)*norm.cdf(-d2) - self.params.S0*norm.cdf(-d1)

    def delta(self, sigma):
        d1 = (np.log(self.params.S0 / self.K) +
              (self.params.r + 0.5*sigma**2)*self.params.T) / (sigma*np.sqrt(self.params.T))
        return norm.cdf(d1) if self.option_type == "call" else norm.cdf(d1)-1

    def implied_vol(self, market_price):
        def f(sigma):
            return self.black_scholes_price(sigma) - market_price
        try:
            return brentq(f, 1e-6, 5)
        except:
            return np.nan

# -----------------------------
# Classe Call et Put
# -----------------------------


class Call(Option):
    option_type = "call"

    def payoff(self, ST):
        return np.maximum(ST - self.K, 0)


class Put(Option):
    option_type = "put"

    def payoff(self, ST):
        return np.maximum(self.K - ST, 0)

# -----------------------------
# Classe pour tracer Payoff
# -----------------------------


class PayoffPlot:
    @staticmethod
    def plot(option):
        ST = np.linspace(0.5*option.params.S0, 1.5*option.params.S0, 100)
        plt.figure(figsize=(8, 5))
        plt.plot(ST, option.payoff(ST))
        plt.title(f"Payoff {option.option_type} Strike={option.K}")
        plt.xlabel("Prix sous-jacent")
        plt.ylabel("Payoff")
        plt.grid(True)
        plt.show()

# -----------------------------
# Classe pour tracer Volatility Smile
# -----------------------------


class SmilePlot:
    @staticmethod
    def plot(strikes, vols, option_type="call"):
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, vols, marker='o')
        plt.title(f"Volatility Smile ({option_type})")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.grid(True)
        plt.show()


# -----------------------------
# Exemple d'utilisation
# -----------------------------
# Lecture CSV
calls_csv = pd.read_csv("data/calls.csv")
puts_csv = pd.read_csv("data/puts.csv")

# Paramètres marché
params = Parameters(S0=170, r=0.05, T=30/365)

# Calcul volatilité implicite pour chaque call
calls_csv['impliedVol'] = calls_csv.apply(
    lambda row: Call(row['strike'], params).implied_vol(row['lastPrice']), axis=1)

# Calcul prix BS et delta
calls_csv['BS_price'] = calls_csv.apply(
    lambda row: Call(row['strike'], params).black_scholes_price(row['impliedVol']), axis=1)
calls_csv['Delta'] = calls_csv.apply(
    lambda row: Call(row['strike'], params).delta(row['impliedVol']), axis=1)

print(calls_csv.head())

# Tracer Volatility Smile
SmilePlot.plot(calls_csv['strike'], calls_csv['impliedVol'], "call")

# Tracer payoff pour un strike choisi
PayoffPlot.plot(Call(170, params))
