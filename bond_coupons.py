# rates_products.py
from dataclasses import dataclass
from typing import List
from yield_curve import YieldCurve
import math


@dataclass
class Bond:
    nominal: float
    coupon_rate: float      # ex 0.05 pour 5% par an
    maturity: float         # en années
    frequency: int = 1      # nombre de coupons par an

    def cashflow_dates(self) -> List[float]:
        n = int(self.maturity * self.frequency)
        return [i / self.frequency for i in range(1, n + 1)]

    def price(self, curve: YieldCurve) -> float:
        dates = self.cashflow_dates()
        c = self.coupon_rate * self.nominal / self.frequency
        price_coupons = sum(c * curve.discount_factor(t) for t in dates)
        price_nominal = self.nominal * curve.discount_factor(self.maturity)
        return price_coupons + price_nominal


def equity_future_price(spot: float, rate: float, T: float, dividend_yield: float = 0.0) -> float:
    return spot * math.exp((rate - dividend_yield) * T)




@dataclass
class InterestRateSwap:
    notional: float
    fixed_rate: float        # taux fixe K
    payment_dates: List[float]  # en années
    year_fraction: float = 1.0  # alpha_i (ex 0.5 si semestriel)

    def price_payer(self, curve: YieldCurve) -> float:
        """
        Prix pour un swap PAYER fixe / RECEVEUR flottant.
        """
        dfs = [curve.discount_factor(t) for t in self.payment_dates]
        df0 = 1.0
        dfn = dfs[-1]
        float_leg = self.notional * (df0 - dfn)
        fixed_leg = self.notional * self.fixed_rate * self.year_fraction * sum(dfs)
        return float_leg - fixed_leg
