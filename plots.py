# plots.py

import numpy as np
import matplotlib.pyplot as plt


class PayoffPlot:
    @staticmethod
    def plot(option):
        S0 = option.params.S0
        ST = np.linspace(0.5 * S0, 1.5 * S0, 200)

        plt.figure(figsize=(8, 5))
        plt.plot(ST, option.payoff(ST))
        plt.title(f"Payoff {option.option_type} - K={option.K}")
        plt.xlabel("Prix du sous-jacent à maturité")
        plt.ylabel("Payoff")
        plt.grid(True)
        plt.show()


class SmilePlot:
    @staticmethod
    def plot(strikes, vols, option_type="call"):
        plt.figure(figsize=(8, 5))
        plt.plot(strikes, vols, marker='o')
        plt.title(f"Volatility Smile ({option_type})")
        plt.xlabel("Strike")
        plt.ylabel("Implied Volatility")
        plt.grid(True)
        plt.show()


class SurfacePlot:
    @staticmethod
    def plot_surface(strikes, maturities, vol_matrix):
        '''
        strikes: liste de K
        maturities: liste de T
        vol_matrix : matrice (len(maturities) × len(strikes))
        '''

        K, T = np.meshgrid(strikes, maturities)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(K, T, vol_matrix, cmap='viridis')
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T")
        ax.set_zlabel("Vol Implicite")
        ax.set_title("Surface de Volatilité")
        plt.show()
