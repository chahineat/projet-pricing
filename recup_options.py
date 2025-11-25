import yfinance as yf
import pandas as pd
import os

ticker = yf.Ticker("AAPL")  # Apple (US)

# Liste des expirations (ex : les 5 premières)
expirations = ticker.options[:5]
print("Expirations disponibles :", expirations)

# Créer dossier data
os.makedirs("data", exist_ok=True)

# Boucle sur les expirations
for exp in expirations:
    options = ticker.option_chain(exp)
    calls = options.calls
    puts = options.puts

    # Sauvegarde CSV pour chaque expiration
    calls.to_csv(f"data/calls_{exp}.csv", index=False)
    puts.to_csv(f"data/puts_{exp}.csv", index=False)

    print(f"Fichiers CSV pour {exp} générés.")

# Affichage exemple pour la première expiration
print(calls.head())
print(puts.head())
