from .discount_factors import DiscountCurve


def forward_rate(curve: DiscountCurve, T1: float, T2: float) -> float:
    """
    Wrapper simple: renvoie le taux forward continu f(T1, T2).
    """
    return curve.forward_rate(T1, T2)


def simple_forward_rate(curve: DiscountCurve, T1: float, T2: float) -> float:
    """
    Taux forward simple (non continu) F_{T1,T2} ~ (R*(T2-T1)).
    On convertit le forward continu en taux simple pour la période.
    """
    f = curve.forward_rate(T1, T2)
    return (pow(2.718281828, f * (T2 - T1)) - 1) / (T2 - T1)
