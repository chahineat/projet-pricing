from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import bisect


@dataclass
class VolPoint:
    """
    Un point de surface de volatilité:
      - T : maturité en années
      - K : strike
      - iv : volatilité implicite (décimal)
    """
    T: float
    K: float
    iv: float


class VolSurface:
    """
    Surface de volatilité implicite discrète, avec interpolation bilinéaire simple.

    On stocke les données sous forme de DataFrame avec colonnes:
      - 'T' (maturité en années)
      - 'K' (strike)
      - 'iv' (implied volatility en décimal)
      - optionnel: 'maturity_str' (pour info)
    """

    def __init__(self, df: pd.DataFrame):
        if not {"T", "K", "iv"}.issubset(df.columns):
            raise ValueError("df doit contenir les colonnes 'T', 'K', 'iv'.")
        df = df.dropna(subset=["T", "K", "iv"]).copy()
        if df.empty:
            raise ValueError("Surface vide.")
        self._df = df
        # liste triée des maturités uniques
        self._Ts = sorted(df["T"].unique())

    @property
    def raw(self) -> pd.DataFrame:
        """DataFrame brut de la surface."""
        return self._df

    @property
    def maturities(self) -> List[float]:
        """Liste triée des maturités disponibles (en années)."""
        return self._Ts

    def strikes_for_T(self, T: float) -> np.ndarray:
        """Renvoie les strikes disponibles pour la maturité la plus proche de T."""
        T_near = self._nearest_T(T)
        sub = self._df[self._df["T"] == T_near]
        return np.sort(sub["K"].unique())

    def smile(self, T: float) -> pd.DataFrame:
        """
        Renvoie un smile (DataFrame strike, iv) pour la maturité la plus proche de T.
        """
        T_near = self._nearest_T(T)
        sub = self._df[self._df["T"] == T_near].copy()
        return sub[["K", "iv"]].sort_values("K")

    def _nearest_T(self, T: float) -> float:
        """
        Trouve la maturité disponible la plus proche de T.
        """
        Ts = self._Ts
        if T <= Ts[0]:
            return Ts[0]
        if T >= Ts[-1]:
            return Ts[-1]
        i = bisect.bisect_left(Ts, T)
        before, after = Ts[i - 1], Ts[i]
        return before if abs(T - before) <= abs(T - after) else after

    def iv_at(self, K: float, T: float) -> float:
        """
        Approximates sigma(K, T) by bilinear interpolation on (T, K).
        If outside the convex hull, performs edge extrapolation.
        """
        Ts = self._Ts
        if T <= Ts[0]:
            T1 = T2 = Ts[0]
        elif T >= Ts[-1]:
            T1 = T2 = Ts[-1]
        else:
            j = bisect.bisect_left(Ts, T)
            T1, T2 = Ts[j - 1], Ts[j]

        iv_T1 = self._iv_at_T_slice(K, T1)
        iv_T2 = self._iv_at_T_slice(K, T2)

        if T1 == T2:
            return iv_T1

        w = (T - T1) / (T2 - T1)
        return iv_T1 * (1 - w) + iv_T2 * w

    def _iv_at_T_slice(self, K: float, T: float) -> float:
        """
        Interpolation 1D en strike pour une maturité fixée.
        """
        sub = self._df[self._df["T"] == T].sort_values("K")
        Ks = sub["K"].to_numpy()
        IVs = sub["iv"].to_numpy()

        if K <= Ks[0]:
            return IVs[0]
        if K >= Ks[-1]:
            return IVs[-1]

        i = bisect.bisect_left(Ks, K)
        K1, K2 = Ks[i - 1], Ks[i]
        iv1, iv2 = IVs[i - 1], IVs[i]

        w = (K - K1) / (K2 - K1)
        return iv1 * (1 - w) + iv2 * w
