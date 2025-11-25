import os
from datetime import date


def equity_snapshot_path(data_dir: str, ticker: str, valuation_date: date) -> str:
    """
    Path standardisé pour un snapshot equity.
    Exemple: data/equity/AAPL_2025-11-24.parquet
    """
    folder = os.path.join(data_dir, "equity")
    os.makedirs(folder, exist_ok=True)
    fname = f"{ticker.upper()}_{valuation_date.isoformat()}.parquet"
    return os.path.join(folder, fname)


def options_snapshot_path(data_dir: str, ticker: str, valuation_date: date) -> str:
    """
    Path pour les snapshots d'options (par exemple plusieurs maturités dans un seul fichier).
    """
    folder = os.path.join(data_dir, "options")
    os.makedirs(folder, exist_ok=True)
    fname = f"{ticker.upper()}_{valuation_date.isoformat()}.parquet"
    return os.path.join(folder, fname)


def rates_snapshot_path(data_dir: str, curve_name: str, valuation_date: date) -> str:
    """
    Path pour une courbe de taux donnée (ex: USD_OIS, USD_SWAP).
    """
    folder = os.path.join(data_dir, "rates")
    os.makedirs(folder, exist_ok=True)
    fname = f"{curve_name}_{valuation_date.isoformat()}.csv"
    return os.path.join(folder, fname)
