import matplotlib.pyplot as plt
from ..vol_smile import VolSmile


def plot_smile(smile: VolSmile, title_prefix: str = "Volatility smile"):
    df = smile.sorted()
    fig, ax = plt.subplots()
    ax.plot(df["K"], df["iv"], marker="o")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f"{title_prefix} (T={smile.T:.3f}y)")
    ax.grid(True)
    return fig
