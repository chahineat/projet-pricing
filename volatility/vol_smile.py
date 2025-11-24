from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .vol_surface import VolSurface


SmileSide = Literal["call", "put", "both"]


@dataclass
class VolSmile:
    """
    Smile de volatilité pour une maturité donnée.
    df doit contenir au moins:
      - 'K' : strike
      - 'iv' : implied vol
    """
    T: float
    df: pd.DataFrame

    def sorted(self) -> pd.DataFrame:
        return self.df.sort_values("K")

    @property
    def strikes(self) -> np.ndarray:
        return self.sorted()["K"].to_numpy()

    @property
    def ivs(self) -> np.ndarray:
        return self.sorted()["iv"].to_numpy()


def smile_from_surface(surface: VolSurface, T: float) -> VolSmile:
    """
    Construis un VolSmile à partir d'une VolSurface pour la maturité la plus
    proche de T.
    """
    smile_df = surface.smile(T)
    # On récupère la vraie T utilisée (nearest)
    T_near = surface._nearest_T(T)
    return VolSmile(T=T_near, df=smile_df)


def smile_from_option_chain(chain_df: pd.DataFrame, T: float) -> VolSmile:
    """
    Construis un VolSmile directement depuis un DataFrame d'options (une maturité).
    chain_df doit contenir au moins:
        - 'strike'
        - 'impliedVolatility'
    """
    if not {"strike", "impliedVolatility"}.issubset(chain_df.columns):
        raise ValueError("chain_df doit contenir 'strike' et 'impliedVolatility'.")
    df = chain_df.copy()
    df = df[df["impliedVolatility"] > 1e-6]
    smile_df = df.rename(columns={"strike": "K", "impliedVolatility": "iv"})[["K", "iv"]]
    return VolSmile(T=T, df=smile_df)
