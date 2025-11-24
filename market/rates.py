from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .base import MarketDataSource
from .config import MarketConfig, DataMode
from .paths import rates_snapshot_path


@dataclass
class RatesConfig:
    curve_name: str  # ex: "USD_TREASURY" ou "USD_OIS"


class RatesMarketData(MarketDataSource):
    """
    Données brutes d'une courbe de taux (maturité, rate).
    C'est la "matière première" pour le bootstrap.
    """

    def __init__(self, config: MarketConfig, rt_config: RatesConfig):
        super().__init__(config)
        self.rt_config = rt_config
        self._df: Optional[pd.DataFrame] = None  # colonnes: maturity, rate

    @property
    def curve_name(self) -> str:
        return self.rt_config.curve_name

    @property
    def raw_curve(self) -> pd.DataFrame:
        if self._df is None:
            # en mode gros projet, tu peux décider que LIVE = pas supporté,
            # et tout passe par des snapshots pré-remplis.
            self.load_snapshot()
        return self._df

    def save_snapshot(self) -> None:
        if self._df is None:
            raise RuntimeError("Aucune donnée de courbe chargée pour sauvegarde.")
        path = rates_snapshot_path(self.config.data_dir, self.curve_name, self.valuation_date)
        self._df.to_csv(path, index=False)

    def load_snapshot(self) -> None:
        path = rates_snapshot_path(self.config.data_dir, self.curve_name, self.valuation_date)
        self._df = pd.read_csv(path)
