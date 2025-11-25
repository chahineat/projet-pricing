# test_surface.py
import matplotlib.pyplot as plt
import glob
import pandas as pd
from parameters import Parameters
from option import Call
from pricer import Pricer
from plots import SurfacePlot

# Paramètres marché
S0 = 170
r = 0.05

# Récupération de tous les fichiers calls CSV
all_calls_files = glob.glob("data/calls_*.csv")

expirations = []
strikes = sorted(pd.read_csv(all_calls_files[0])['strike'])
vol_matrix = []

for file in all_calls_files:
    df = pd.read_csv(file)

    # Date d'expiration en années
    exp_date = file.split("_")[-1].replace(".csv", "")
    T = (pd.to_datetime(exp_date) - pd.Timestamp.today()).days / 365
    expirations.append(T)

    params = Parameters(S0=S0, r=r, T=T)
    vols = []
    for K in strikes:
        row = df[df['strike'] == K]
        if not row.empty:
            market_price = row['lastPrice'].values[0]
            opt = Call(K, params)
            sigma = Pricer.implied_vol(opt, market_price)
            vols.append(sigma)
        else:
            vols.append(float('nan'))

    vol_matrix.append(vols)

vol_matrix = pd.DataFrame(vol_matrix, index=expirations, columns=strikes)

# Tracé de la surface
SurfacePlot.plot_surface(strikes, expirations, vol_matrix.values)
