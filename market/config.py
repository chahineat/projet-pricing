from dataclasses import dataclass
from datetime import date
from enum import Enum


class DataMode(str, Enum):
    LIVE = "live"        # on interroge les APIs (yfinance, etc.)
    SNAPSHOT = "snapshot"  # on lit des fichiers locaux (CSV/Parquet)


@dataclass
class MarketConfig:
    """
    Configuration globale pour les données de marché.
    """
    valuation_date: date
    mode: DataMode = DataMode.SNAPSHOT  # par défaut : reproductible
    currency: str = "USD"
    data_dir: str = "data"              # racine des données locales
