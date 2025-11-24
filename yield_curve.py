# yield_curve.py
from dataclasses import dataclass
from typing import List
import bisect
import math


@dataclass
class CurvePoint:
    maturity: float  # en années
    rate: float      # en décimal, ex 0.03 pour 3%


class YieldCurve:
    """
    Courbe de taux simple avec interpolation linéaire sur les taux.
    On travaille en taux continus pour avoir DF = exp(-r T).
    """

    def __init__(self, points: List[CurvePoint]):
        # on trie par maturité
        self.points = sorted(points, key=lambda p: p.maturity)
        self.maturities = [p.maturity for p in self.points]
        self.rates = [p.rate for p in self.points]

    def get_rate(self, T: float) -> float:
        """
        Retourne le taux interpolé pour maturité T (en années).
        """
        if T <= self.maturities[0]:
            return self.rates[0]
        if T >= self.maturities[-1]:
            return self.rates[-1]

        i = bisect.bisect_left(self.maturities, T)
        t1, t2 = self.maturities[i - 1], self.maturities[i]
        r1, r2 = self.rates[i - 1], self.rates[i]
        # interpolation linéaire
        w = (T - t1) / (t2 - t1)
        return r1 * (1 - w) + r2 * w

    def discount_factor(self, T: float) -> float:
        """
        DF(0,T) = exp(-r(T) * T) avec r(T) taux interpolé.
        """
        r = self.get_rate(T)
        return math.exp(-r * T)

'''
pts = [
    CurvePoint(0.5, 0.03),
    CurvePoint(1.0, 0.032),
    CurvePoint(2.0, 0.035),
    CurvePoint(5.0, 0.04),
    CurvePoint(10.0, 0.045),
]
curve = YieldCurve(pts)
print(curve.discount_factor(3.0))'''
