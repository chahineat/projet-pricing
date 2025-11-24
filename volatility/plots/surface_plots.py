import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import numpy as np

from ..vol_surface import VolSurface


def plot_vol_surface(surface: VolSurface, n_T: int = 20, n_K: int = 50):
    Ts = np.linspace(min(surface.maturities), max(surface.maturities), n_T)
    all_K = surface.raw["K"].to_numpy()
    K_min, K_max = np.min(all_K), np.max(all_K)
    Ks = np.linspace(K_min, K_max, n_K)

    TT, KK = np.meshgrid(Ts, Ks, indexing="ij")
    IV = np.zeros_like(TT)

    for i in range(TT.shape[0]):
        for j in range(TT.shape[1]):
            IV[i, j] = surface.iv_at(KK[i, j], TT[i, j])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(KK, TT, IV, linewidth=0, antialiased=True)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Maturity T (years)")
    ax.set_zlabel("Implied Vol")
    ax.set_title("Volatility Surface")
    return fig
