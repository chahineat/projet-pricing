from dataclasses import dataclass
from typing import Optional

import datetime as dt

import pandas as pd
import yfinance as yf



DEFAULT_TICKER = "AAPL"
DEFAULT_RF_SERIES = "DGS1MO"  # Taux sans risque : US Treasury 1 mois (source FRED)


@dataclass
class MarketDataConfig:
    """
    Objet de configuration pour MarketData.
    """
    ticker: str = DEFAULT_TICKER
    rf_series: str = DEFAULT_RF_SERIES
    history_years: int = 1  # nombre d'années de données historiques pour le spot


class MarketData:
    """
    Classe centralisant l'accès aux données de marché :
      - prix spot
      - historique de prix
      - taux sans risque
      - chaîne d'options + volatilités implicites
    """

    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()

        # Objet yfinance pour le ticker
        self._yf_ticker = yf.Ticker(self.config.ticker)

        # Téléchargement de l'historique de prix (spot)
        end = dt.datetime.today()
        start = end - dt.timedelta(days=365 * self.config.history_years)
        self.history = self._yf_ticker.history(start=start, end=end)

        if self.history.empty:
            raise ValueError(f"Aucune donnée historique trouvée pour le ticker {self.config.ticker}")

    # ------------------------------------------------------------------
    # 1) Prix spot
    # ------------------------------------------------------------------
    @property
    def spot(self) -> float:
        """
        Retourne le prix spot (dernier prix de clôture disponible).
        """
        return float(self.history["Close"].iloc[-1])

    # ------------------------------------------------------------------
    # 2) Taux sans risque
    # ------------------------------------------------------------------
    def get_risk_free_rate(self):
        rate = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1]
        return float(rate) / 100

    # ------------------------------------------------------------------
    # 3) Chaîne d'options et volatilités implicites
    # ------------------------------------------------------------------
    def get_available_maturities(self):
        """
        Liste les maturités (dates d'expiration) disponibles pour les options.
        """
        return self._yf_ticker.options

    def get_option_chain(self, maturity: str):
        """
        Récupère la chaîne d'options (calls + puts) pour une maturité donnée.
        Paramètre:
            maturity : string au format 'YYYY-MM-DD'

        Retourne:
            calls, puts : deux DataFrame pandas
        """
        chain = self._yf_ticker.option_chain(maturity)
        return chain.calls, chain.puts

    def get_call_iv_for_strike(self, maturity: str, strike: float) -> float:
        """
        Retourne la volatilité implicite (IV) d'un call pour un strike donné.

        Si le strike exact n'existe pas dans la chaîne, on prend le strike le plus proche.
        """
        calls, _ = self.get_option_chain(maturity)

        if calls.empty:
            raise ValueError(f"Aucune option call trouvée pour la maturité {maturity}")

        # On sélectionne le strike le plus proche
        idx = (calls["strike"] - strike).abs().idxmin()
        iv = float(calls.loc[idx, "impliedVolatility"])
        return iv

    # ------------------------------------------------------------------
    # 4) Fonctions utilitaires de visualisation (optionnel)
    # ------------------------------------------------------------------
    def get_return_series(self) -> pd.Series:
        """
        Retourne la série de rendements journaliers log (pour analyse / plots).
        """
        close = self.history["Close"]
        log_returns = (close / close.shift(1)).apply(lambda x: pd.NA if pd.isna(x) else pd.np.log(x))
        return log_returns.dropna()


# ----------------------------------------------------------------------
# Exemple d'utilisation (pour test / notebook)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    config = MarketDataConfig(ticker="AAPL", rf_series="DGS1MO", history_years=1)
    mkt = MarketData(config)

    print(f"Ticker : {config.ticker}")
    print(f"Spot   : {mkt.spot:.2f} USD")

    r = mkt.get_risk_free_rate()
    print(f"Taux sans risque (1M) : {100 * r:.2f} %")

    maturities = mkt.get_available_maturities()
    print("Maturités disponibles :", maturities)

    # On prend une maturité un peu plus lointaine: 3e
    if len(maturities) >= 4:
        maturity = maturities[3]
    else:
        maturity = maturities[0]

    print(f"Maturité utilisée pour l'IV : {maturity}")

    # Exemple : IV pour un strike proche du spot
    iv = mkt.get_call_iv_for_strike(maturity, strike=mkt.spot)
    print(f"\nVolatilité implicite (IV) ~ at-the-money : {100 * iv:.2f} %")
