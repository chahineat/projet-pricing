from dataclasses import dataclass
from typing import List

from .discount_factors import DiscountCurve


@dataclass
class InterestRateSwap:
    notional: float
    fixed_rate: float           # taux fixe du swap (en décimal)
    payment_times: List[float]  # dates de paiement en années
    year_fraction: float = 1.0  # alpha (1.0 = annuel, 0.5 = semestriel)

    def par_rate(self, curve: DiscountCurve) -> float:
        """
        Calcule le taux fixe 'par' du swap (valeur initiale NPV=0).
        """
        dfs = [curve.df(t) for t in self.payment_times]
        denom = self.year_fraction * sum(dfs)
        if denom == 0:
            raise ValueError("Somme des DF nulle, problème de courbe.")
        df0 = 1.0
        dfn = dfs[-1]
        float_leg = df0 - dfn
        return float_leg / denom

    def npv_payer(self, curve: DiscountCurve) -> float:
        """
        NPV d'un swap PAYER fixe / RECEVEUR flottant.
        """
        dfs = [curve.df(t) for t in self.payment_times]
        df0 = 1.0
        dfn = dfs[-1]
        float_leg = self.notional * (df0 - dfn)
        fixed_leg = self.notional * self.fixed_rate * self.year_fraction * sum(dfs)
        return float_leg - fixed_leg
