import yfinance as yf
import pandas as pd
import os

ticker = yf.Ticker("AAPL")  # Apple (US)

print(ticker.options)        # Devrait afficher une liste de dates d'expiration

# Choisir la première date d'expiration
expiration = '2025-11-28'

# Récupérer les options Call et Put
options = ticker.option_chain(expiration)
calls = options.calls
puts = options.puts

# Créer un dossier data pour sauvegarder les fichiers
os.makedirs("data", exist_ok=True)

# Sauvegarder en CSV
calls.to_csv("data/calls.csv", index=False)
puts.to_csv("data/puts.csv", index=False)

print("Fichiers CSV générés dans le dossier 'data'.")

# Afficher les premières lignes
print(calls.head())
print(puts.head())
