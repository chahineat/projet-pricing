# rates/bootstrap_curve.py
import math
import pandas as pd

from .discount_factors import DiscountCurve


def bootstrap_from_zero_rates(
    df: pd.DataFrame,
    col_maturity: str = "maturity",
    col_rate: str = "rate",
    rate_is_continuous: bool = True,
) -> DiscountCurve:
    """
    Construit une DiscountCurve à partir de taux zéro-coupon.

    Paramètres
    ----------
    df : DataFrame
        Doit contenir au minimum :
          - une colonne de maturités (en années)
          - une colonne de taux (en décimal, ex 0.03)

    col_maturity : str
        Nom de la colonne pour les maturités (par défaut 'maturity').

    col_rate : str
        Nom de la colonne pour les taux (par défaut 'rate').

    rate_is_continuous : bool
        - True  : on considère que les taux sont déjà en convention continue :
                 DF(0,T) = exp(-r * T)
        - False : on considère que les taux sont des taux simples annuels :
                 (1 + r * T) → convertis en taux continus.

    Retour
    ------
    DiscountCurve
        Objet représentant DF(0,T) pour toutes les maturités.
    """

    if col_maturity not in df.columns or col_rate not in df.columns:
        raise ValueError(
            f"Le DataFrame doit contenir les colonnes '{col_maturity}' et '{col_rate}'. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df_sorted = df.sort_values(col_maturity)
    maturities = df_sorted[col_maturity].to_list()
    rates = df_sorted[col_rate].to_list()

    dfs = []
    for T, r in zip(maturities, rates):
        if T <= 0:
            raise ValueError("Les maturités doivent être > 0")
        if rate_is_continuous:
            # DF = e^{-r T}
            DF = math.exp(-r * T)
        else:
            # r = taux simple annuel → on convertit en continu
            # (1 + r*T) = e^{r_cont * T}  =>  r_cont = ln(1 + r*T)/T
            r_cont = math.log(1.0 + r * T) / T
            DF = math.exp(-r_cont * T)
        dfs.append(DF)

    return DiscountCurve(maturities=maturities, dfs=dfs)
