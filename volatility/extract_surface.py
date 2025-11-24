from dataclasses import dataclass
from datetime import date
from typing import Optional, List

import pandas as pd

from market import MarketConfig, DataMode, OptionChainMarketData, OptionChainConfig, EquityMarketData, EquityConfig


@dataclass
class SurfaceExtractionConfig:
    ticker: str
    max_maturities: Optional[int] = 5
    min_iv: float = 1e-4
    use_calls: bool = True  # on peut aussi combiner calls + puts si besoin


def _date_diff_in_years(d1: date, d2: date) -> float:
    return (d2 - d1).days / 365.0


def extract_vol_surface(
    mkt_config: MarketConfig,
    surf_conf: SurfaceExtractionConfig,
    eq_conf: Optional[EquityConfig] = None,
) -> pd.DataFrame:
    """
    Construit un DataFrame 'surface' avec colonnes:
        - maturity_str (YYYY-MM-DD)
        - T (en années, temps jusqu'à l'échéance)
        - K (strike)
        - iv (implied vol)
        - type (call/put)

    À partir des données d'options Yahoo via OptionChainMarketData.
    """
    eq_conf = eq_conf or EquityConfig(ticker=surf_conf.ticker)
    equity_mkt = EquityMarketData(mkt_config, eq_conf)
    _ = equity_mkt.spot  # force le chargement/snapshot si besoin

    opt_mkt = OptionChainMarketData(
        mkt_config, OptionChainConfig(ticker=surf_conf.ticker, max_maturities=surf_conf.max_maturities)
    )

    rows = []
    val_date = mkt_config.valuation_date

    for mat_str in opt_mkt.maturities:
        calls, puts = opt_mkt.get_chain(mat_str)
        exp_date = date.fromisoformat(mat_str)
        T = _date_diff_in_years(val_date, exp_date)
        if T <= 0:
            continue

        if surf_conf.use_calls:
            df_use = calls
            opt_type = "call"
        else:
            df_use = puts
            opt_type = "put"

        df_use = df_use[df_use["impliedVolatility"] > surf_conf.min_iv].copy()
        if df_use.empty:
            continue

        for _, row in df_use.iterrows():
            rows.append(
                {
                    "maturity_str": mat_str,
                    "T": T,
                    "K": float(row["strike"]),
                    "iv": float(row["impliedVolatility"]),
                    "type": opt_type,
                }
            )

    surface_df = pd.DataFrame(rows)
    if surface_df.empty:
        raise ValueError("Surface vide après extraction. Vérifier les filtres/paramètres.")
    return surface_df
