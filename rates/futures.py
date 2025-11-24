import math

from .discount_factors import DiscountCurve
from .forward_rates import forward_rate


def interest_rate_future_forward(curve: DiscountCurve, T1: float, T2: float) -> float:
    """
    Taux forward implicite, utilisé comme sous-jacent d'un future de taux.
    En pratique, les futures Eurodollar sont quotés ~ 100 - 100*L,
    mais ici on renvoie juste le forward continu f(T1,T2).
    """
    return forward_rate(curve, T1, T2)


def equity_future_price(spot: float, rate: float, T: float, dividend_yield: float = 0.0) -> float:
    """
    Prix théorique d'un future sur une action ou un indice:
        F0 = S0 * exp((r - q) * T)
    """
    return spot * math.exp((rate - dividend_yield) * T)
