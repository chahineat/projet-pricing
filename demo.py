# demo.py
from Market_data import MarketData
from yield_curve import YieldCurve, CurvePoint
from bond_coupons import Bond, InterestRateSwap, equity_future_price
from equity_options_bs import BlackScholesModel
from plots import plot_call_payoff, plot_iv_smile


def main():
    # 1) Market data
    mkt = MarketData()
    S0 = mkt.spot
    r = mkt.get_risk_free_rate()
    maturities = mkt.get_available_maturities()
    maturity = maturities[3]  # on prend une maturité un peu lointaine
    iv_atm = mkt.get_call_iv_for_strike(maturity, strike=S0)

    print(f"Spot AAPL = {S0:.2f}, r = {100*r:.2f} %, maturity={maturity}, IV_ATM={100*iv_atm:.2f} %")

    # 2) Courbe de taux
    curve = YieldCurve([
        CurvePoint(0.5, 0.03),
        CurvePoint(1.0, 0.032),
        CurvePoint(2.0, 0.035),
        CurvePoint(5.0, 0.04),
        CurvePoint(10.0, 0.045),
    ])

    # 3) Bond
    bond = Bond(nominal=100.0, coupon_rate=0.04, maturity=5.0, frequency=1)
    bond_price = bond.price(curve)
    print(f"Prix du bond 5 ans 4% = {bond_price:.2f}")

    # 4) Swap
    swap = InterestRateSwap(
        notional=100.0,
        fixed_rate=0.04,
        payment_dates=[1, 2, 3, 4, 5],
        year_fraction=1.0
    )
    swap_price = swap.price_payer(curve)
    print(f"Prix du payer swap 5 ans 4% = {swap_price:.4f}")

    # 5) Future equity (1 an)
    future = equity_future_price(S0, r, T=1.0)
    print(f"Future 1 an sur AAPL ≈ {future:.2f}")

    # 6) Option equity Black–Scholes
    # approx T pour la maturité choisie : disons 0.8 ans pour l'exemple
    T = 0.8
    model = BlackScholesModel(spot=S0, rate=r, volatility=iv_atm)
    K = S0
    call_price = model.call_price(K, T)
    put_price = model.put_price(K, T)
    print(f"Call ATM (K=S0) ~ {call_price:.2f}, Put ATM ~ {put_price:.2f}")

    # 7) Graphiques
    plot_call_payoff(K)
    plot_iv_smile(maturity_index=3)


if __name__ == "__main__":
    main()
