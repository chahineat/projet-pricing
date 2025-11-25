from .discount_factors import DiscountCurve


def fra_forward_rate(curve: DiscountCurve, T1: float, T2: float) -> float:
    """
    Taux forward (continu) entre T1 et T2; on peut le convertir en simple si besoin.
    """
    return curve.forward_rate(T1, T2)


def fra_price(notional: float, K: float, T1: float, T2: float, curve: DiscountCurve) -> float:
    """
    Prix (valeur actuelle) d'un FRA qui paye (L - K) * (T2 - T1) * N à T2,
    avec L = taux forward implicite entre T1 et T2.

    Hypothèse: taux simples sur la période, actualisation avec DF(0, T2).
    """
    tau = T2 - T1
    F = fra_forward_rate(curve, T1, T2)  # taux continu
    # on convertit en taux simple pour la période:
    L_simple = (pow(2.718281828, F * tau) - 1) / tau
    payoff_T2 = (L_simple - K) * tau * notional
    return payoff_T2 * curve.df(T2)
