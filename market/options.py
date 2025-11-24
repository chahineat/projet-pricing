from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Dict

import pandas as pd
import yfinance as yf

from .base import MarketDataSource
from .config import MarketConfig, DataMode
from .paths import options_snapshot_path


@dataclass
class OptionChainConfig:
    ticker: str
    max_maturities: Optional[int] = None


class OptionChainMarketData(MarketDataSource):
    """
    Données de marché pour les options d'un sous-jacent (toutes maturités).
    """

    def __init__(self, config: MarketConfig, opt_config: OptionChainConfig):
        super().__init__(config)
        self.opt_config = opt_config
        # structure: { maturity_str: {"calls": df, "puts": df} }
        self._chains: Dict[str, Dict[str, pd.DataFrame]] = {}

    @property
    def ticker(self) -> str:
        return self.opt_config.ticker.upper()

    @property
    def maturities(self) -> List[str]:
        if not self._chains:
            if self.config.mode == DataMode.SNAPSHOT:
                try:
                    self.load_snapshot()
                except FileNotFoundError:
                    self._download_all_chains()
            else:
                self._download_all_chains()
        return list(self._chains.keys())

    def get_chain(self, maturity: str) -> (pd.DataFrame, pd.DataFrame):
        if not self._chains:
            if self.config.mode == DataMode.SNAPSHOT:
                try:
                    self.load_snapshot()
                except FileNotFoundError:
                    self._download_all_chains()
            else:
                self._download_all_chains()
        data = self._chains.get(maturity)
        if data is None:
            raise ValueError(f"Maturité {maturity} non trouvée")
        return data["calls"], data["puts"]


    def _download_all_chains(self) -> None:
        t = yf.Ticker(self.ticker)
        all_mats = t.options
        if self.opt_config.max_maturities is not None:
            all_mats = all_mats[: self.opt_config.max_maturities]

        chains: Dict[str, Dict[str, pd.DataFrame]] = {}
        for mat in all_mats:
            chain = t.option_chain(mat)
            calls = chain.calls
            puts = chain.puts
            # nettoyage simple IV
            calls = calls[calls["impliedVolatility"] > 1e-6]
            puts = puts[puts["impliedVolatility"] > 1e-6]
            chains[mat] = {"calls": calls, "puts": puts}

        self._chains = chains

    def save_snapshot(self) -> None:
        if not self._chains:
            self._download_all_chains()
        path = options_snapshot_path(self.config.data_dir, self.ticker, self.valuation_date)
        rows = []
        for mat, data in self._chains.items():
            for typ in ("calls", "puts"):
                df = data[typ].copy()
                df["maturity"] = mat
                df["type"] = typ
                rows.append(df)
        big_df = pd.concat(rows, ignore_index=True)
        big_df.to_parquet(path)

    def load_snapshot(self) -> None:
        path = options_snapshot_path(self.config.data_dir, self.ticker, self.valuation_date)
        big_df = pd.read_parquet(path)
        chains: Dict[str, Dict[str, pd.DataFrame]] = {}
        for mat, grp in big_df.groupby("maturity"):
            calls = grp[grp["type"] == "calls"].drop(columns=["maturity", "type"])
            puts = grp[grp["type"] == "puts"].drop(columns=["maturity", "type"])
            chains[mat] = {"calls": calls, "puts": puts}
        self._chains = chains

    def smile_for_maturity(self, maturity: str) -> pd.DataFrame:
        calls, _ = self.get_chain(maturity)
        smile = calls[["strike", "impliedVolatility"]].copy()
        smile = smile.sort_values("strike")
        return smile
