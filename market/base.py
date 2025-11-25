from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from .config import MarketConfig


@dataclass
class MarketDataSource(ABC):
    """
    Classe abstraite pour toute source de données de marché.
    """
    config: MarketConfig

    @property
    def valuation_date(self) -> date:
        return self.config.valuation_date

    @abstractmethod
    def save_snapshot(self) -> None:
        """
        Sauvegarde les données de marché associées dans un fichier local
        (CSV/Parquet/…).
        """
        ...

    @abstractmethod
    def load_snapshot(self) -> None:
        """
        Recharge les données de marché depuis un fichier local.
        """
        ...
