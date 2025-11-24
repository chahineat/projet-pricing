from dataclasses import dataclass
from typing import List

from .discount_factors import DiscountCurve


@dataclass
class CouponBond:
    nominal: float
    coupon_rate: float      # ex: 0.04 = 4% par an
    maturity: float         # en années
    frequency: int = 1      # nb de coupons par an (1 = annuel, 2 = semestriel)

    def cashflow_times(self) -> List[float]:
        """
        Renvoie la liste des dates de cash-flows (en années).
        """
        n = int(self.maturity * self.frequency)
        return [i / self.frequency for i in range(1, n + 1)]

    def price(self, curve: DiscountCurve) -> float:
        """
        Prix du bond comme somme des coupons actualisés + nominal actualisé.
        """
        times = self.cashflow_times()
        c = self.coupon_rate * self.nominal / self.frequency
        pv_coupons = sum(c * curve.df(t) for t in times)
        pv_nominal = self.nominal * curve.df(self.maturity)
        return pv_coupons + pv_nominal
