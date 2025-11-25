from .config import MarketConfig
from .equity import EquityMarketData, EquityConfig
from .options import OptionChainMarketData, OptionChainConfig
from .rates import RatesMarketData, RatesConfig


def build_full_snapshot(config: MarketConfig):
    """
    Extrait (en mode LIVE) puis sauvegarde (en snapshot) toutes les
    données marché nécessaires à un scénario de valorisation.
    """
    # Ex. sur AAPL + USD_TREASURY
    eq = EquityMarketData(config, EquityConfig(ticker="AAPL"))
    eq.save_snapshot()

    opt = OptionChainMarketData(config, OptionChainConfig(ticker="AAPL", max_maturities=5))
    opt.save_snapshot()

    # Pour la courbe de taux, souvent tu rempliras le CSV à la main.
    # Mais si tu as un script d'extraction, tu peux aussi l'appeler ici.


def load_all_market(config: MarketConfig):
    """
    Recharge toutes les market data nécessaires à partir des snapshots.
    """
    eq = EquityMarketData(config, EquityConfig(ticker="AAPL"))
    _ = eq.history  # force le load

    opt = OptionChainMarketData(config, OptionChainConfig(ticker="AAPL", max_maturities=5))
    _ = opt.maturities

    # idem pour la courbe de taux…
    return {"equity": eq, "options": opt}
