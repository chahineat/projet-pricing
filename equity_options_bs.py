# equity_option_bs.py
import math
from dataclasses import dataclass

SQRT_2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """
    CDF de la loi normale standard N(0,1) via erf.
    """
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


@dataclass
class BlackScholesModel:
    spot: float
    rate: float      # r
    volatility: float  # sigma
    dividend_yield: float = 0.0  # q

    def d1_d2(self, strike: float, maturity: float):
        S, r, q, sigma, T = self.spot, self.rate, self.dividend_yield, self.volatility, maturity
        if sigma <= 0 or T <= 0:
            raise ValueError("sigma et T doivent être > 0")
        d1 = (math.log(S / strike) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def call_price(self, strike: float, maturity: float) -> float:
        d1, d2 = self.d1_d2(strike, maturity)
        S, r, q, K, T = self.spot, self.rate, self.dividend_yield, strike, maturity
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

    def put_price(self, strike: float, maturity: float) -> float:
        d1, d2 = self.d1_d2(strike, maturity)
        S, r, q, K, T = self.spot, self.rate, self.dividend_yield, strike, maturity
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)
