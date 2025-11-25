# market/rates.py
from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
import yfinance as yf

from .base import MarketDataSource
from .config import MarketConfig, DataMode
from .paths import rates_snapshot_path


@dataclass
class RatesConfig:
    curve_name: str  # ex: "USD_ZERO"


class RatesMarketData(MarketDataSource):
    """
    Market data pour une courbe de taux simple.

    Deux modes :
      - LIVE : récupère quelques taux US Treasuries depuis Yahoo Finance
      - SNAPSHOT : lit un CSV local data/rates/<curve_name>_<date>.csv
    """

    def __init__(self, config: MarketConfig, rates_config: RatesConfig):
        super().__init__(config)
        self.rates_config = rates_config
        self.curve_name = rates_config.curve_name
        self._df: Optional[pd.DataFrame] = None

    # ------------ API publique ------------

    @property
    def raw_curve(self) -> pd.DataFrame:
        """
        Renvoie un DataFrame avec au moins:
            - 'maturity' (en années)
            - 'rate'    (en décimal, taux zéro continu ou simple)
        """
        if self._df is None:
            if self.config.mode == DataMode.LIVE:
                self._download_live_from_yahoo()
            else:
                self.load_snapshot()
        return self._df

    # ------------ LIVE (Yahoo) ------------

    def _download_live_from_yahoo(self) -> None:
        """
        Construit une pseudo-courbe USD à partir de quelques taux US Treasury
        via Yahoo Finance :

            ^IRX : 13-week T-Bill  ~ 0.25 an
            ^FVX : 5Y Treasury     ~ 5 ans
            ^TNX : 10Y Treasury    ~ 10 ans
            ^TYX : 30Y Treasury    ~ 30 ans

        Les tickers Yahoo sont donnés en '% * 10' (par ex. 45.0 = 4.5%),
        on convertit en décimal: (y / 10) / 100.
        """
        tickers: Dict[float, str] = {
            0.25: "^IRX",
            5.0: "^FVX",
            10.0: "^TNX",
            30.0: "^TYX",
        }

        data = yf.download(
            list(tickers.values()),
            period="5d",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if "Adj Close" in data:
            last = data["Adj Close"].iloc[-1]
        else:
            last = data["Close"].iloc[-1]

        rows = []
        for T, tic in tickers.items():
            y_raw = float(last[tic])
            rate = (y_raw / 10.0) / 100.0  # ex: 45.0 -> 4.5% -> 0.045
            rows.append({"maturity": T, "rate": rate})

        self._df = pd.DataFrame(rows)

    # ------------ SNAPSHOT ------------

    def save_snapshot(self) -> None:
        """
        Sauvegarde la courbe actuelle dans data/rates/<name>_<date>.csv
        (utilisable ensuite en mode SNAPSHOT).
        """
        if self._df is None:
            raise RuntimeError("Aucune donnée de courbe chargée pour sauvegarde.")
        path = rates_snapshot_path(self.config.data_dir, self.curve_name, self.valuation_date)
        self._df.to_csv(path, index=False)

    def load_snapshot(self) -> None:
        """
        Recharge une courbe sauvegardée au format CSV.
        """
        path = rates_snapshot_path(self.config.data_dir, self.curve_name, self.valuation_date)
        self._df = pd.read_csv(path)
