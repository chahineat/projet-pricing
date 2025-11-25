import math
import bisect
from dataclasses import dataclass
from typing import List


@dataclass
class DiscountCurve:
    """
    Courbe de discount paramétrée par une liste de maturités T_i et
    de facteurs d'actualisation DF_i.

    Hypothèses:
      - T_i en années, strictement croissants
      - DF_i = DF(0, T_i), 0 < DF_i <= 1
    """
    maturities: List[float]  # T_i
    dfs: List[float]         # DF_i

    def __post_init__(self):
        if len(self.maturities) != len(self.dfs):
            raise ValueError("maturities et dfs doivent avoir la même longueur")
        if any(t <= 0 for t in self.maturities):
            raise ValueError("Toutes les maturités doivent être > 0")
        if sorted(self.maturities) != list(self.maturities):
            raise ValueError("Les maturités doivent être strictement croissantes")

    def df(self, T: float) -> float:
        """
        Discount factor DF(0,T) via interpolation linéaire sur ln(DF).
        """
        if T <= self.maturities[0]:
            return self.dfs[0]
        if T >= self.maturities[-1]:
            return self.dfs[-1]

        i = bisect.bisect_left(self.maturities, T)
        T1, T2 = self.maturities[i - 1], self.maturities[i]
        DF1, DF2 = self.dfs[i - 1], self.dfs[i]

        # interpolation linéaire sur log(DF)
        logDF1, logDF2 = math.log(DF1), math.log(DF2)
        w = (T - T1) / (T2 - T1)
        logDF = logDF1 * (1 - w) + logDF2 * w
        return math.exp(logDF)

    def zero_rate(self, T: float) -> float:
        """
        Taux zéro-coupon continu r(T) tel que DF(0,T) = exp(-r(T) * T).
        """
        if T <= 0:
            raise ValueError("T doit être > 0 pour un zero rate.")
        DF = self.df(T)
        return -math.log(DF) / T

    def forward_rate(self, T1: float, T2: float) -> float:
        """
        Taux forward continu entre T1 et T2:
            f(T1, T2) tel que DF(0,T1)/DF(0,T2) = exp(-f*(T2-T1)).
        """
        if T2 <= T1:
            raise ValueError("On doit avoir T2 > T1 pour un forward rate.")
        DF1 = self.df(T1)
        DF2 = self.df(T2)
        return (math.log(DF1) - math.log(DF2)) / (T2 - T1)
