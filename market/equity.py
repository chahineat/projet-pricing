from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import MarketDataSource
from .config import MarketConfig, DataMode
from .paths import equity_snapshot_path


@dataclass
class EquityConfig:
    ticker: str
    history_years: int = 2
    use_adjusted_close: bool = True


class EquityMarketData(MarketDataSource):
    """
    Données de marché pour une action (spot + historique).
    """

    def __init__(self, config: MarketConfig, eq_config: EquityConfig):
        super().__init__(config)
        self.eq_config = eq_config
        self._history: Optional[pd.DataFrame] = None

    # --------- Propriétés principales ---------

    @property
    def ticker(self) -> str:
        return self.eq_config.ticker.upper()

    @property
    def history(self) -> pd.DataFrame:
        """
        Historique des prix (DataFrame yfinance).
        Si mode = SNAPSHOT mais le fichier n'existe pas, on tombe sur LIVE.
        """
        if self._history is None:
            if self.config.mode == DataMode.SNAPSHOT:
                try:
                    self.load_snapshot()
                except FileNotFoundError:
                    # fallback automatique vers yfinance
                    self._download_history()
            else:
                self._download_history()
        return self._history

    @property
    def spot(self) -> float:
        """
        Spot = dernier prix de clôture.
        On essaie d'abord 'Adj Close', sinon 'Close', sinon la dernière colonne.
        """
        hist = self.history

        for col in ["Adj Close", "Close", "close"]:
            if col in hist.columns:
                return float(hist[col].iloc[-1])

        # fallback de secours : dernière colonne de la dernière ligne
        return float(hist.iloc[-1, -1])

    # --------- Méthodes internes ---------

    def _download_history(self) -> None:
        """
        Télécharge l'historique depuis yfinance autour de la valuation_date.
        """
        end = self.valuation_date + timedelta(days=1)
        start = self.valuation_date - timedelta(days=365 * self.eq_config.history_years)
        ticker = yf.Ticker(self.ticker)
        hist = ticker.history(start=start, end=end)
        if hist.empty:
            raise ValueError(f"Aucune donnée historique pour {self.ticker}")
        self._history = hist

    # --------- Snapshot I/O ---------

    def save_snapshot(self) -> None:
        """
        Sauvegarde l'historique complet dans un fichier Parquet.
        """
        if self._history is None:
            self._download_history()
        path = equity_snapshot_path(self.config.data_dir, self.ticker, self.valuation_date)
        self._history.to_parquet(path)

    def load_snapshot(self) -> None:
        """
        Recharge l'historique depuis un snapshot Parquet.
        """
        path = equity_snapshot_path(self.config.data_dir, self.ticker, self.valuation_date)
        self._history = pd.read_parquet(path)

    # --------- Fonctions utilitaires ---------

    def log_returns(self) -> pd.Series:
        col = "Adj Close" if self.eq_config.use_adjusted_close else "Close"
        prices = self.history[col]
        rets = (prices / prices.shift(1)).apply(lambda x: pd.NA if pd.isna(x) else pd.np.log(x))
        return rets.dropna()
