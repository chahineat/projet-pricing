# test.py

import pandas as pd
import numpy as np
from parameters import Parameters
from option import Call
from pricer import Pricer
from plots import PayoffPlot, SmilePlot

# Charger les données
calls = pd.read_csv("data/calls.csv")

# Paramètres marché
params = Parameters(S0=170, r=0.05, T=30/365)

# ---- Fonctions sécurisées (évite NaN, prix nuls, divisions impossibles) ----


def safe_implied_vol(row):
    price = row.get("lastPrice", np.nan)
    if np.isnan(price) or price <= 0:
        return np.nan
    opt = Call(row["strike"], params)
    return Pricer.implied_vol(opt, price)


def safe_bs_price(row):
    if np.isnan(row["impliedVol"]):
        return np.nan
    opt = Call(row["strike"], params)
    return Pricer.bs_price(opt, row["impliedVol"])


def safe_delta(row):
    if np.isnan(row["impliedVol"]):
        return np.nan
    opt = Call(row["strike"], params)
    return Pricer.delta(opt, row["impliedVol"])


# ---- Calcul vol implicite ----
calls["impliedVol"] = calls.apply(safe_implied_vol, axis=1)

# ---- Calcul prix théorique + delta ----
calls["BS_price"] = calls.apply(safe_bs_price, axis=1)
calls["Delta"] = calls.apply(safe_delta, axis=1)

print(calls[["strike", "lastPrice", "impliedVol", "BS_price", "Delta"]].head())

# ---- Graphiques ----
SmilePlot.plot(calls["strike"], calls["impliedVol"], option_type="call")
PayoffPlot.plot(Call(170, params))
