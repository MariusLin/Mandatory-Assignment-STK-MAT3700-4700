import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
def calculate_black_scholes(r, s, K, S, T):
    """
    Calculate the price of a European call option using the Black-Scholes formula
    from the lecture notes.
    Parameters:
    -----------
    r : float
        The risk-free interest rate.
    s : float
        The volatility of the underlying asset.
    K : float
        The strike price of the option.
    S : float
        The current price of the underlying asset.
    T : float
        Time to maturity of the option in years.
    Returns:
    --------
    float
        The theoretical price of the European call option based on the Black-Scholes model.
    """
    d_1 = (np.log(S/K) + (r+(s**2)/2)*T)/(s*np.sqrt(T))
    d_2 = d_1 - s*np.sqrt(T)
    phi_1 = norm.cdf(d_1)
    phi_2 = norm.cdf(d_2)
    return S*phi_1 - K*phi_2*np.exp(-r*T)


def implied_volatility(S, K, T, r, market_price):
    """
    Calculate the implied volatility of a European call option using the Black-Scholes model.
    Parameters:
    -----------
    S : float
        The current price of the underlying asset.
    K : float
        The strike price of the option.
    T : float
        Time to maturity of the option in years.
    r : float
        The risk-free interest rate .
    market_price : float
        The observed market price of the option.
    Returns:
    --------
    float
        The implied volatility that equates the Black-Scholes price of the option to
        the observed market price.
    """
    def price_diff(s):
        return calculate_black_scholes(r, s, K , S, T) - market_price

    # brentq is used to find where the difference between theoretical and market price euqates 0
    return brentq(price_diff, -1, 1)

def task_2_a(sigma, T):
    """
    Generate and print LaTeX-formatted Black-Scholes option prices for
    multiple volatilities and times to maturity.

    Parameters:
    -----------
    sigma : list of float
        A list of volatilities for which Black-Scholes prices will be calculated.
    T : list of float
        A list of times to maturity for which Black-Scholes prices will be calculated.

    Returns:
    --------
    None
        This function prints the generated LaTeX-formatted option prices for each
         combination of volatility and time to maturity.
    """
    for s in sigma:
        print (s)
        for time in T:
            print (time)
            latex_list = []
            latex = ""
            for strike in K:
                if strike == 130 and s== 0.15 and time == 1/12:
                    latex += str(
                        np.round(calculate_black_scholes(0.026, s, strike, 100, time), 10)) + "\% \cdot S_0"
                elif strike == 130:
                    latex += str(np.round(calculate_black_scholes(0.026, s, strike, 100, time), 5)) + "\% \cdot S_0"
                else:
                    latex += str(np.round(calculate_black_scholes(0.026, s, strike, 100, time), 5)) + "\% \cdot S_0 &"
            latex_list.append(latex)
            print(latex_list[0])


def task_2_b():
    """
        Plot implied volatilities of a European call option against strike prices.

        This function calculates the implied volatility for a given list of market
        prices and strike prices using the Black-Scholes model.
        It then generates a scatter plot of implied volatility
        against strike prices and saves the plot as an image file.

        Parameters:
        -----------
        None
        Returns:
        --------
        None
            The function generates and saves a plot of implied volatilities vs. strike prices.
        """
    market_prices = [27.85 , 24.1 , 20.3, 16.75,  14.05 , 11.33 , 8.9 , 6.93 , 5.15]
    strikes = [200  ,205  , 210, 215 , 220 , 225 , 230 , 235 , 240]
    imp_volatility = [implied_volatility(221.19, strike, 100/366, 0.026, market_price) for strike, market_price in zip(strikes, market_prices)]
    plt.scatter(strikes, imp_volatility, marker = "x", label = "Implied volatilities against strikes")
    plt.plot(strikes, imp_volatility, linestyle='--', color='red', label = "Trend")
    plt.xlabel("Strikes")
    plt.ylabel("Implied volatility")
    plt.title("Implied volatility against strike")
    plt.legend()
    plt.savefig('implied_volatilities.png', dpi=300, bbox_inches='tight')

# Uncomment this for seeing the results:
"""
T =[1/12, 1/4, 1/2]
K = [100, 90, 110, 70, 130]
sigma = [0.15,0.3,0.45]
task_2_a(sigma, T)
task_2_b()
"""
