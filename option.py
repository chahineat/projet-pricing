# option.py

import numpy as np


class Option:
    """
    Classe abstraite d'option européenne.
    Ne contient PAS le pricing : uniquement K, params et payoff.
    """

    def __init__(self, K, parameters):
        self.K = float(K)
        self.params = parameters  # contient S0, r, T

    # méthode redéfinie dans Call/Put
    def payoff(self, ST):
        raise NotImplementedError("payoff must be implemented in subclasses")


class Call(Option):
    option_type = "call"

    def payoff(self, ST):
        return np.maximum(ST - self.K, 0)


class Put(Option):
    option_type = "put"

    def payoff(self, ST):
        return np.maximum(self.K - ST, 0)
