# parameters.py

class Parameters:
    """
    Classe qui regroupe les paramètres globaux :
    - S0 : prix spot du sous-jacent
    - r  : taux sans risque
    - T  : maturité en années
    """

    def __init__(self, S0, r, T):
        self.S0 = float(S0)
        self.r = float(r)
        self.T = float(T)

    def __repr__(self):
        return f"Parameters(S0={self.S0}, r={self.r}, T={self.T})"
