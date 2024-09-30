import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from matplotlib.lines import Line2D

def estimates(period, interval, ticker):
    """
    Estimate the average return and volatility for a given stock over a
    specified period and interval.

    Parameters:
    -----------
    period : str
        The total time span of historical stock data to retrieve. Possible values include:
        ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].
    interval : str
        The data frequency for stock price retrieval.
        Common values include '1d' (daily), '1wk' (weekly), or '1mo' (monthly).
    ticker : str
        The ticker symbol of the stock for which the estimates are to be calculated.

    Returns:
    --------
    tuple
        A tuple containing:
        - return_estimate : float
            The estimated average return of the stock over the specified period.
        - volatility_estimate : float
            The estimated volatility of the stock over the specified period.
        - returns : numpy.ndarray
            The array of calculated daily returns for the stock.

    Notes:
    ------
    - As the data used for the calculations ends at 11 September 2024, there might be
    differences in these results compared to those presented in the handed-in assignment
    """
    stock = yf.Ticker(ticker)
    prices = stock.history(period=period, interval=interval)["Close"].values
    returns = [(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
    N = len(returns)
    returns = np.array(returns)
    return_estimate = (1 / N) * np.sum(returns)
    volatility_estimate = np.sqrt((1 / (N - 1)) * np.sum((returns - return_estimate) ** 2))
    return return_estimate, volatility_estimate, returns

def estimates_print(period, interval, ticker):
    """
    Analogous to the estimates-function. The only difference is that it returns
    the annualised estimates
    """
    stock = yf.Ticker(ticker)
    prices = stock.history(period=period, interval=interval)["Close"].values
    returns = [(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices)-1)]
    N = len(returns)
    returns = np.array(returns)
    return_estimate = (1 / N) * np.sum(returns)
    volatility_estimate = np.sqrt((1 / (N - 1)) * np.sum((returns - return_estimate) ** 2))
    if period == "1y": #Annualise according to period
        return_estimate *= 252
        volatility_estimate *= np.sqrt(252)
    else:
        return_estimate *= 52
        volatility_estimate *= np.sqrt(52)
    return return_estimate, volatility_estimate, returns


def plot_normal_distributions(period, interval, ticker):
    """
        Plot the empirical and fitted normal distributions of stock returns over a
        specified period and interval.

        This function generates a probability density plot comparing the empirical
        distribution of stock returns with a fitted normal distribution based on estimated mean
        and standard deviation. The plot is saved as an image.

        Parameters:
        -----------
        period : str
            The total time span of historical stock data to retrieve. Possible values include:
            ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'].
        interval : str
            The data frequency for stock price retrieval.
            Common values include '1d' (daily), '1wk' (weekly), or '1mo' (monthly).
        ticker : str
            The ticker symbol of the stock for which the distributions will be plotted.

        Returns:
        --------
        None
            The function generates and saves a plot comparing the empirical and
            fitted normal distributions of stock returns.

        """
    plt.clf()
    mu, std, returns = estimates(period, interval, ticker)
    sns.kdeplot(returns, color='g', label='Empirical Normal Distribution')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal Distribution')
    title = "Empirical and fitted normal distribution"
    plt.title(title)
    plt.ylabel('Probability Density')
    # This makes it possible to add the type of data (weekly, daily) to the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if period == "1y":
        new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
        new_labels = ['Daily data from one year']
    else:
        new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
        new_labels = ['Weekly data from two years']
    handles.extend([new_handle1])
    labels.extend(new_labels)
    current_legend = plt.gca().get_legend()
    if current_legend:
        current_legend.remove()
    plt.legend(handles=handles, labels=labels, loc='upper right')

    plt.savefig(f'plot_{ticker}_{period}.png', dpi=300, bbox_inches='tight')

def estimate_covariance(return_estimate_1, returns_1, return_estimate_2, returns_2):
    """
    Estimate the covariance between two sets of stock returns.

    This function calculates the covariance between two return series using
    the sample covariance formula.

    Parameters:
    -----------
    return_estimate_1 : float
        The estimated average return of the first stock.
    returns_1 : numpy.ndarray
        An array of historical returns for the first stock.
    return_estimate_2 : float
        The estimated average return of the second stock.
    returns_2 : numpy.ndarray
        An array of historical returns for the second stock.

    Returns:
    --------
    float
        The estimated covariance between the two return series.

    Notes:
    ------
    - As all the data is obtained from American stocks, the length of the return-vectors are equal
    and, thus, do not need to be aligned
    """
    return 1/(len(returns_1)-1)*(np.matmul((returns_1-return_estimate_1).T,returns_2-return_estimate_2))
def estimate_correlation (return_estimate_1, returns_1, return_estimate_2, returns_2,
                          volatility_estimate_1, volatility_estimate_2):
    """
    Estimate the correlation between two sets of stock returns.

    This function calculates the correlation coefficient using the covariance
    of the two return series and their respective volatility estimates.

    Parameters:
    -----------
    return_estimate_1 : float
        The estimated average return of the first stock.
    returns_1 : numpy.ndarray
        An array of historical returns for the first stock.
    return_estimate_2 : float
        The estimated average return of the second stock.
    returns_2 : numpy.ndarray
        An array of historical returns for the second stock.
    volatility_estimate_1 : float
        The estimated volatility for the first stock.
    volatility_estimate_2 : float
        The estimated volatility for the second stock.

    Returns:
    --------
    float
        The estimated correlation coefficient between the two return series,
        rounded to five decimal places.
    """
    covariance = estimate_covariance(return_estimate_1, returns_1, return_estimate_2, returns_2)
    return round(covariance/(volatility_estimate_1 * volatility_estimate_2),5)
def covariance_matrix(re_1, r1, re_2, r2, re_3, r3, re_4, r4, re_5, r5, number):
    """
    Compute the covariance matrix for up to five sets of stock returns.

    This function calculates the covariance matrix based on the provided estimated
    returns and historical returns for up to five different assets.
    The matrix is symmetric and captures the relationships between the returns of the assets.

    Parameters:
    -----------
    re_1 : float
        The estimated average return of the first stock.
    r1 : numpy.ndarray
        An array of historical returns for the first stock.
    re_2 : float
        The estimated average return of the second stock.
    r2 : numpy.ndarray
        An array of historical returns for the second stock.
    re_3 : float
        The estimated average return of the third stock.
    r3 : numpy.ndarray
        An array of historical returns for the third stock.
    re_4 : float
        The estimated average return of the fourth stock.
    r4 : numpy.ndarray
        An array of historical returns for the fourth stock.
    re_5 : float
        The estimated average return of the fifth stock (only used if `number` is 5).
    r5 : numpy.ndarray
        An array of historical returns for the fifth stock (only used if `number` is 5).
    number : int
        The number of assets for which to compute the covariance matrix.
        Acceptable values are 4 or 5.

    Returns:
    --------
    numpy.ndarray
        A (4x4) or (5x5) covariance matrix depending on the `number` parameter.
    """
    if number == 5:
        cov_matrix = np.zeros((5,5))
        # first row/column
        cov_matrix[0,0]= estimate_covariance(re_1, r1, re_1, r1)
        cov_matrix[0,1]= estimate_covariance(re_1, r1, re_2, r2)
        cov_matrix[1,0] = estimate_covariance(re_1, r1, re_2, r2)
        cov_matrix[0,2]= estimate_covariance(re_1, r1, re_3, r3)
        cov_matrix[2,0] = estimate_covariance(re_1, r1, re_3, r3)
        cov_matrix[0,3] = estimate_covariance(re_1, r1, re_4, r4)
        cov_matrix[3,0] = estimate_covariance(re_1, r1, re_4, r4)
        cov_matrix[0,4] = estimate_covariance(re_1, r1, re_5, r5)
        cov_matrix[4,0] = estimate_covariance(re_1, r1, re_5, r5)
        # second row/column
        cov_matrix[1,1] = estimate_covariance(re_2, r2, re_2, r2)
        cov_matrix[1,2] = estimate_covariance(re_2, r2, re_3, r3)
        cov_matrix[2,1] = estimate_covariance(re_2, r2, re_3, r3)
        cov_matrix[1,3] = estimate_covariance(re_2, r2, re_4, r4)
        cov_matrix[3,1] = estimate_covariance(re_2, r2, re_4, r4)
        cov_matrix[1,4] = estimate_covariance(re_2, r2, re_5, r5)
        cov_matrix[4,1] = estimate_covariance(re_2, r2, re_5, r5)
        # third row/column
        cov_matrix[2,2]= estimate_covariance(re_3, r3, re_3, r3)
        cov_matrix[2,3]= estimate_covariance(re_3, r3, re_4, r4)
        cov_matrix[3,2] = estimate_covariance(re_3, r3, re_4, r4)
        cov_matrix[2,4]= estimate_covariance(re_3, r3, re_5, r5)
        cov_matrix[4,2] = estimate_covariance(re_3, r3, re_5, r5)
        #fourth row/column
        cov_matrix[3,3] = estimate_covariance(re_4, r4, re_4, r4)
        cov_matrix[3,4] = estimate_covariance(re_4, r4, re_5, r5)
        cov_matrix[4,3] = estimate_covariance(re_4, r4, re_5, r5)
        #fifth row/column
        cov_matrix[4,4] = estimate_covariance(re_5,r5, re_5, r5)
        np.set_printoptions(precision=7, suppress=True)
        return cov_matrix
    else:
        cov_matrix = np.zeros((4,4))
        # first row/column
        cov_matrix[0,0]= estimate_covariance(re_1, r1, re_1, r1)
        cov_matrix[0,1]= estimate_covariance(re_1, r1, re_2, r2)
        cov_matrix[1,0] = estimate_covariance(re_1, r1, re_2, r2)
        cov_matrix[0,2]= estimate_covariance(re_1, r1, re_3, r3)
        cov_matrix[2,0] = estimate_covariance(re_1, r1, re_3, r3)
        cov_matrix[0,3] = estimate_covariance(re_1, r1, re_4, r4)
        cov_matrix[3,0] = estimate_covariance(re_1, r1, re_4, r4)
        # second row/column
        cov_matrix[1,1] = estimate_covariance(re_2, r2, re_2, r2)
        cov_matrix[1,2] = estimate_covariance(re_2, r2, re_3, r3)
        cov_matrix[2,1] = estimate_covariance(re_2, r2, re_3, r3)
        cov_matrix[1,3] = estimate_covariance(re_2, r2, re_4, r4)
        cov_matrix[3,1] = estimate_covariance(re_2, r2, re_4, r4)
        # third row/column
        cov_matrix[2,2]= estimate_covariance(re_3, r3, re_3, r3)
        cov_matrix[2,3]= estimate_covariance(re_3, r3, re_4, r4)
        cov_matrix[3,2] = estimate_covariance(re_3, r3, re_4, r4)
        #fourth row/column
        cov_matrix[3,3] = estimate_covariance(re_4, r4, re_4, r4)
        np.set_printoptions(precision=7, suppress=True)
        return cov_matrix

def calculate_efficient_frontier(C, r, sigma, names,  period, number):
    """
    Calculate and plot the efficient frontier for a given set of assets.

    This function computes the efficient frontier and the minimum variance portfolio
    for a set of assets based on their return and covariance data.
    It visualizes the relationship between risk and return using a plot and returns key
    statistics of the minimum variance portfolio.

    Parameters:
    -----------
    C : numpy.ndarray
        The covariance matrix of the asset returns.

    r : numpy.ndarray
        A vector of expected returns for the assets.

    sigma : numpy.ndarray
        A vector of the standard deviations for each asset.

    names : list of str
        A list of names corresponding to each asset, used for labeling in the plot.

    period : str
        A string indicating the time period for the data. This affects the range
        of risk values in the plot.
        Expected values are "1y" for one year of data or "2y" for two years of data.

    number : int
        The number of assets to consider in the calculation. Should be either 4 or 5.

    Returns:
    --------
    tuple
        A tuple containing three elements:
            - r_min : float
                The expected return of the minimum variance portfolio.
            - sigma_min : float
                The standard deviation of the minimum variance portfolio.
            - w_min : numpy.ndarray
                The weights of the assets in the minimum variance portfolio.
    """
    # calcualte r given s
    def calculate_r(s):
        return (a/c)+np.sqrt((b-(a**2)/c)*(s**2-1/c))
    plt.clf()
    if number == 5:
        # calculates a, b and c
        ones = np.ones(number)
        help_1 = np.linalg.solve(C,r)
        help_2 = np.linalg.solve(C,ones)
        a = np.matmul(ones.T, help_1)
        b = np.matmul(r.T, help_1)
        c = np.matmul(ones.T, help_2)
        r_min = a/c
        w_min = 1/c * help_2
        sigma_min = 1 / np.sqrt(c)
        # for plotting reasons, we discriminate the periods
        if period=="1y":
            sigma_vector = np.arange(sigma_min, 4*sigma_min, step = sigma_min/50)
            r_vector = [calculate_r(s) for s in sigma_vector]
        else:
            sigma_vector = np.arange(sigma_min, 6*sigma_min, step = sigma_min/50)
            r_vector = [calculate_r(s) for s in sigma_vector]

        plt.plot(sigma_vector, r_vector)
        plt.scatter(sigma_min,r_min, marker = "x", color = "red", label="Minimum variance portfolio")
        # add the single stocks
        for i in range(number):
            plt.scatter(sigma[i], r[i], marker = "x", label = names[i], s= 100)
        plt.xlabel("Risk σ")
        plt.ylabel("Return r")
        plt.title("Efficient portfolio frontier and minimum variance portfolio")
        # see above, manipulating the legend
        handles, labels = plt.gca().get_legend_handles_labels()
        if period == "1y":
            new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
            new_labels = ['Daily data from one year']
        else:
            new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
            new_labels = ['Weekly data from two years']
        handles.extend([new_handle1])
        labels.extend(new_labels)
        current_legend = plt.gca().get_legend()
        if current_legend:
            current_legend.remove()
        plt.legend(handles=handles, labels=labels, loc='upper left')
        plt.savefig(f'plot_efficientFrontier_five_{period}.png', dpi=300, bbox_inches='tight')
        return round(r_min,5), round(sigma_min,5), w_min
    else:
        #analogous to the other case
        ones = np.ones(number)
        help_1 = np.linalg.solve(C, r)
        help_2 = np.linalg.solve(C, ones)
        a = np.matmul(ones.T, help_1)
        b = np.matmul(r.T, help_1)
        c = np.matmul(ones.T, help_2)
        r_min = a / c
        sigma_min = 1 / np.sqrt(c)
        w_min = 1 / c * help_2
        sigma_vector = np.arange(sigma_min, 4 * sigma_min, step=sigma_min / 50)
        r_vector = np.array([calculate_r(s) for s in sigma_vector])

        plt.plot(sigma_vector, r_vector)
        plt.xlabel("Risk σ")
        plt.ylabel("Return r")
        plt.title("Efficient portfolio frontier and minimum variance portfolio")
        handles, labels = plt.gca().get_legend_handles_labels()
        if period == "1y":
            new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
            new_labels = ['Daily data from one year']
        else:
            new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
            new_labels = ['Weekly data from two years']
        handles.extend([new_handle1])
        labels.extend(new_labels)
        current_legend = plt.gca().get_legend()
        if current_legend:
            current_legend.remove()
        plt.legend(handles=handles, labels=labels, loc='upper left')
        plt.savefig(f'plot_efficientFrontier_four_{period}.png', dpi=300, bbox_inches='tight')
        return round(r_min, 5), round(sigma_min, 5), w_min


def compare_efficient_frontiers(C_five, r_five, C_four, r_four,  period):
    """
        Compare the efficient frontiers of two portfolios, one with five assets
        and another with four assets.

        This function calculates and visualizes the efficient frontiers for two different
        sets of assets, allowing for a comparison between the portfolios.
        It plots the relationship between risk and return for both portfolios and saves
        the plot as a PNG file.

        Parameters:
        -----------
        C_five : numpy.ndarray
            The covariance matrix of the returns for the five-asset portfolio.

        r_five : numpy.ndarray
            A vector of expected returns for the five assets in the portfolio.

        C_four : numpy.ndarray
            The covariance matrix of the returns for the four-asset portfolio.

        r_four : numpy.ndarray
            A vector of expected returns for the four assets in the portfolio.

        period : str
            A string indicating the time period for the data used. This influences the range
            of risk values in the plot.
            Expected values are "1y" for one year of data or "2y" for two years of data.

        Returns:
        --------
        None
            This function does not return any values but saves a plot of the efficient
            frontiers as a PNG file.

        """
    # as the a,b and c are different these are also input
    def calculate_r(s,a,b,c):
        return a / c + np.sqrt((b - a ** 2 / c) * (s ** 2 - 1 / c))
    #calculate a, b and c for five assets
    ones_five = np.ones(5)
    help_1_five = np.linalg.solve(C_five, r_five)
    help_2_five = np.linalg.solve(C_five, ones_five)
    a_five = np.matmul(ones_five.T, help_1_five)
    b_five = np.matmul(r_five.T, help_1_five)
    c_five = np.matmul(ones_five.T, help_2_five)
    sigma_min_five = np.sqrt(1/c_five)
    #calculate a, b and c for four assets
    ones_four = np.ones(4)
    help_1_four = np.linalg.solve(C_four, r_four)
    help_2_four = np.linalg.solve(C_four, ones_four)
    a_four = np.matmul(ones_four.T, help_1_four)
    b_four = np.matmul(r_four.T, help_1_four)
    c_four = np.matmul(ones_four.T, help_2_four)
    sigma_min_four = np.sqrt(1 /c_four)
    #for aesthetic reasosn discriminate periods
    if period == "1y":
        sigma_vector_five = np.arange(sigma_min_five, 4 * sigma_min_five, step=sigma_min_five / 50)
        sigma_vector_four = np.arange(sigma_min_four, 4 * sigma_min_four, step=sigma_min_four / 50)
        r_vector_five = [calculate_r(sigma, a_five, b_five, c_five) for sigma in sigma_vector_five]
        r_vector_four = [calculate_r(sigma, a_four, b_four, c_four) for sigma in sigma_vector_four]

    else:
        sigma_vector_five = np.arange(sigma_min_five, 7 * sigma_min_five, step=sigma_min_five / 50)
        sigma_vector_four = np.arange(sigma_min_four, 7 * sigma_min_four, step=sigma_min_four / 50)
        r_vector_five = [calculate_r(sigma, a_five, b_five, c_five) for sigma in sigma_vector_five]
        r_vector_four = [calculate_r(sigma, a_four, b_four, c_four) for sigma in sigma_vector_four]

    plt.plot(sigma_vector_five, r_vector_five, label ="EPF for all five assets")
    plt.plot(sigma_vector_four, r_vector_four, label="EPF for remaining four assets")
    plt.xlabel("Risk σ")
    plt.ylabel("Return r")
    plt.title("Efficient portfolio frontier and minimum variance portfolio")
    # see above, manipulating the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if period == "1y":
        new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
        new_labels = ['Daily data from one year']
    else:
         new_handle1 = Line2D([0], [0], color='none', marker='o', markersize=0)
         new_labels = ['Weekly data from two years']
    handles.extend([new_handle1])
    labels.extend(new_labels)
    current_legend = plt.gca().get_legend()
    if current_legend:
        current_legend.remove()
    plt.legend(handles=handles, labels=labels, loc='upper left')
    plt.savefig(f'plot_efficientFrontier_comparison_{period}.png', dpi=300,
                bbox_inches='tight')


# Initialise all the variables
"""
tickers = ["LMT", "AAPL", "F", "K", "NU"]
names_five = ["Lockheed Martin", "Apple", "Ford Motor Company", "Kellanova", "Nu Holdings"]
names_four = ["Lockheed Martin", "Apple", "Ford Motor Company", "Kellanova"]
re_1_1y, ve_1_1y, r1_1y = estimates("1y", "1d", "LMT")
re_2_1y, ve_2_1y, r2_1y = estimates("1y", "1d", "AAPL")
re_3_1y, ve_3_1y, r3_1y = estimates("1y", "1d", "F")
re_4_1y, ve_4_1y, r4_1y = estimates("1y", "1d", "K")
re_5_1y, ve_5_1y, r5_1y = estimates("1y", "1d", "NU")
re_1_2y, ve_1_2y, r1_2y = estimates("2y", "1wk", "LMT")
re_2_2y, ve_2_2y, r2_2y = estimates("2y", "1wk", "AAPL")
re_3_2y, ve_3_2y, r3_2y = estimates("2y", "1wk", "F")
re_4_2y, ve_4_2y, r4_2y = estimates("2y", "1wk", "K")
re_5_2y, ve_5_2y, r5_2y = estimates("2y", "1wk", "NU")
cov_matrix_1y_five = covariance_matrix(re_1_1y, r1_1y, re_2_1y, r2_1y, re_3_1y, r3_1y, re_4_1y,
r4_1y, re_5_1y, r5_1y,5)
cov_matrix_2y_five = covariance_matrix(re_1_2y, r1_2y, re_2_2y, r2_2y, re_3_2y, r3_2y,
re_4_2y, r4_2y, re_5_2y, r5_2y,5)
r_hat_1y_five = np.array([re_1_1y, re_2_1y, re_3_1y, re_4_1y, re_5_1y])
r_hat_2y_five = np.array([re_1_2y, re_2_2y, re_3_2y, re_4_2y, re_5_2y])
sigma_hat_1y_five = np.array([ve_1_1y, ve_2_1y, ve_3_1y, ve_4_1y, ve_5_1y])
sigma_hat_2y_five = np.array([ve_1_2y, ve_2_2y, ve_3_2y, ve_4_2y, ve_5_2y])
cov_matrix_1y_four = covariance_matrix(re_1_1y, r1_1y, re_2_1y, r2_1y, re_3_1y, r3_1y,
re_4_1y, r4_1y, re_5_1y, r5_1y,4)
cov_matrix_2y_four = covariance_matrix(re_1_2y, r1_2y, re_2_2y, r2_2y, re_3_2y, r3_2y,
re_4_2y, r4_2y, re_5_2y, r5_2y,4)
r_hat_1y_four = np.array([re_1_1y, re_2_1y, re_3_1y, re_4_1y])
r_hat_2y_four = np.array([re_1_2y, re_2_2y, re_3_2y, re_4_2y])
sigma_hat_1y_four = np.array([ve_1_1y, ve_2_1y, ve_3_1y, ve_4_1y])
sigma_hat_2y_four = np.array([ve_1_2y, ve_2_2y, ve_3_2y, ve_4_2y])
"""
#uncomment the initializing for using these functions
def task_a_values():
    """
    Calculate and print estimated returns and volatility for a list of stock tickers
    over two different time periods (1 year and 2 years) using weekly data intervals.

    This function iterates over a predefined list of stock tickers, retrieves estimated returns
    and volatility for each ticker using the `estimates_print` function, and prints the results
    rounded to five decimal places. It performs the calculations for both a 1-year and a 2-year
    period, displaying the results consecutively.

    Parameters:
    -----------
    None
        This function does not take any parameters but relies on a predefined list of stock
        tickers stored in the variable `tickers`.

    Returns:
    --------
    None
        This function does not return any values. It prints the estimated returns and volatility
        for each ticker directly to the console.
    """
    for ticker in tickers:
        ests = estimates_print("1y", "1wk", ticker)
        print (ticker)
        print (np.round(ests[:2],5))

    for ticker in tickers:
        ests = estimates_print("2y", "1wk", ticker)
        print(ticker)
        print(np.round(ests[:2],5))


def task_a_plots():
    """
    Generate and save normal distribution plots for a list of stock tickers over specified time
    periods using different data intervals.

    This function iterates over a predefined list of stock tickers and creates normal distribution
    plots for each ticker. It first generates plots using daily data for the past year and then
    generates plots using weekly data for the past two years. Each plot is saved as a PNG file with
    an appropriate filename indicating the ticker and time period.

    Parameters:
    -----------
    None
        This function does not take any parameters but relies on a predefined list of stock
        tickers stored in the variable `tickers`.

    Returns:
    --------
    None
        This function does not return any values. It generates and saves normal distribution plots
        for each ticker directly to the filesystem.
    """
    for ticker in tickers:
        plot_normal_distributions("1y", "1d", ticker)
    for ticker in tickers:
        plot_normal_distributions("2y", "1wk", ticker)



def task_b_correlations_1y ():
    """
    Calculate and print the correlation coefficients between pairs of asset returns over a
    one-year period.

    This function computes the correlation between the returns of various pairs of assets
    using the `estimate_correlation` function. It covers all possible combinations of asset
    return estimates for five assets over a one-year time frame. The results are printed
    directly to the console.

    The following correlations are calculated:
    - Between asset 1 and assets 2, 3, 4, and 5
    - Between asset 2 and assets 3, 4, and 5
    - Between asset 3 and assets 4 and 5
    - Between asset 4 and asset 5
    These are all cases because the covariance, and thus the correltaion, is symmetric

    Parameters:
    -----------
    None
        This function does not take any parameters but relies on predefined variables that
        store return estimates and volatility for the assets over the one-year period.

    Returns:
    --------
    None
        This function does not return any values. It outputs the calculated correlation
        coefficients to the console.
    """
    print(estimate_correlation(re_1_1y, r1_1y, re_2_1y, r2_1y, ve_1_1y, ve_2_1y))
    print(estimate_correlation(re_1_1y, r1_1y, re_3_1y, r3_1y, ve_1_1y, ve_3_1y))
    print(estimate_correlation(re_1_1y, r1_1y, re_4_1y, r4_1y, ve_1_1y, ve_4_1y))
    print(estimate_correlation(re_1_1y, r1_1y, re_5_1y, r5_1y, ve_1_1y, ve_5_1y))
    print(estimate_correlation(re_2_1y, r2_1y, re_3_1y, r3_1y, ve_2_1y, ve_3_1y))
    print(estimate_correlation(re_2_1y, r2_1y, re_4_1y, r4_1y, ve_2_1y, ve_4_1y))
    print(estimate_correlation(re_2_1y, r2_1y, re_5_1y, r5_1y, ve_2_1y, ve_5_1y))
    print(estimate_correlation(re_3_1y, r3_1y, re_4_1y, r4_1y, ve_4_1y, ve_5_1y))
    print(estimate_correlation(re_3_1y, r3_1y, re_5_1y, r5_1y, ve_3_1y, ve_5_1y))
    print(estimate_correlation(re_4_1y, r4_1y, re_5_1y, r5_1y, ve_4_1y, ve_5_1y))

def task_b_correlations_2y ():
    """
    Analogous to the case for one year, only for two years now
    """
    print(estimate_correlation(re_1_2y, r1_2y, re_2_2y, r2_2y, ve_1_2y, ve_2_2y))
    print(estimate_correlation(re_1_2y, r1_2y, re_3_2y, r3_2y, ve_1_2y, ve_3_2y))
    print(estimate_correlation(re_1_2y, r1_2y, re_4_2y, r4_2y, ve_1_2y, ve_4_2y))
    print(estimate_correlation(re_1_2y, r1_2y, re_5_2y, r5_2y, ve_1_2y, ve_5_2y))
    print(estimate_correlation(re_2_2y, r2_2y, re_3_2y, r3_2y, ve_2_2y, ve_3_2y))
    print(estimate_correlation(re_2_2y, r2_2y, re_4_2y, r4_2y, ve_2_2y, ve_4_2y))
    print(estimate_correlation(re_2_2y, r2_2y, re_5_2y, r5_2y, ve_2_2y, ve_5_2y))
    print(estimate_correlation(re_3_2y, r3_2y, re_4_2y, r4_2y, ve_4_2y, ve_5_2y))
    print(estimate_correlation(re_3_2y, r3_2y, re_5_2y, r5_2y, ve_3_2y, ve_5_2y))
    print(estimate_correlation(re_4_2y, r4_2y, re_5_2y, r5_2y, ve_4_2y, ve_5_2y))

def task_b_covariance_1y():
    """
    Generate and print LaTeX code for a covariance matrix of asset returns over a one-year period.

    This function formats the covariance matrix stored in `cov_matrix_1y_five` into LaTeX syntax,
    enabling users to easily include it in LaTeX documents. Each element of the matrix is rounded
    to five decimal places for clarity.

    The function constructs a LaTeX representation of the covariance matrix using a nested loop
    to iterate over the rows of the matrix. It joins the elements of each row with an ampersand (&)
    and ends each row with a double backslash (\\\\). Finally, it prints the entire matrix enclosed
    within LaTeX's matrix environment.

    Parameters:
    -----------
    None
        This function does not take any parameters but relies on the global variable `
        cov_matrix_1y_five`.

    Returns:
    --------
    None
        This function does not return any values. It directly prints the generated
        LaTeX code to the console.
    """
    latex_code = "\\begin{bmatrix}\n"
    for row in cov_matrix_1y_five:
        latex_code += " & ".join(map(lambda x: f"{x:.5f}", row)) + " \\\\\n"
    latex_code += "\\end{bmatrix}"
    print(latex_code)

def task_b_covariance_2y():
    """
    Analogous to the case for one year
    """
    latex_code = "\\begin{bmatrix}\n"
    for row in cov_matrix_2y_five:
        latex_code += " & ".join(map(lambda x: f"{x:.5f}", row)) + " \\\\\n"
    latex_code += "\\end{bmatrix}"
    print(latex_code)

def task_c_1y():
    """
    Calculate and display the efficient frontier for a portfolio of five assets over a one-year
    period.

    This function invokes the `calculate_efficient_frontier` function using the covariance matrix,
    expected returns, and estimated volatility of five assets, along with their names. It specifies
    the analysis period as one year and indicates that five assets are being considered.

    Parameters:
    -----------
    None
        This function does not take any parameters but relies on several global variables:
        - `cov_matrix_1y_five`: Covariance matrix of asset returns over a one-year period.
        - `r_hat_1y_five`: Expected returns of the five assets over a one-year period.
        - `sigma_hat_1y_five`: Estimated volatilities of the five assets over a one-year period.
        - `names_five`: List of names corresponding to the five assets.

    Returns:
    --------
    None
        This function does not return any values. It directly prints the results of the efficient
        frontier calculation.
    """
    print(calculate_efficient_frontier(cov_matrix_1y_five, r_hat_1y_five, sigma_hat_1y_five,
                                       names_five,"1y",5))
def task_c_2y():
    """
    Analogous to the case of 1y, only over a period of two years
    """
    print(calculate_efficient_frontier(cov_matrix_2y_five, r_hat_2y_five, sigma_hat_2y_five,
                                       names_five, "2y",5))

def task_d_1y():
    """
    Calculate and display the efficient frontier for a portfolio of four assets over a
    one-year period.

    This function calls the `calculate_efficient_frontier` function using the covariance matrix,
    expected returns, and estimated volatility of four assets, along with their names. The analysis
    period is specified as one year, and the function indicates that four assets are being considered.


    Parameters:
    -----------
    None
        This function does not accept any parameters, but it relies on several global variables:
        - `cov_matrix_1y_four`: Covariance matrix of asset returns over a one-year period.
        - `r_hat_1y_four`: Expected returns of the four assets over a one-year period.
        - `sigma_hat_1y_four`: Estimated volatilities of the four assets over a one-year period.
        - `names_four`: List of names corresponding to the four assets.

    Returns:
    --------
    None
        This function does not return any values. It directly prints the results
        of the efficient frontier calculation.
    """
    print(calculate_efficient_frontier(cov_matrix_1y_four, r_hat_1y_four, sigma_hat_1y_four,
                                       names_four,"1y",4))
def task_d_2y():
    """
    Analogous to the case for one year
    """
    print(calculate_efficient_frontier(cov_matrix_2y_four, r_hat_2y_four, sigma_hat_2y_four,
                                       names_four, "2y",4))

def task_d_compare_1y():
    """
    Compare the efficient frontiers for a portfolio of five assets versus four assets
    over a one-year period.

    This function calls the `compare_efficient_frontiers` function, passing the covariance matrices
    and expected returns for both the five-asset portfolio and the four-asset portfolio.
    The analysis is conducted over a one-year period, and the results are visualized
    to illustrate the differences between the two portfolios' efficient frontiers.

    Parameters:
    -----------
    None
        This function does not accept any parameters, but it relies on several global variables:
        - `cov_matrix_1y_five`: Covariance matrix of asset returns for the five-asset portfolio over
         a one-year period.
        - `r_hat_1y_five`: Expected returns of the five assets over a one-year period.
        - `cov_matrix_1y_four`: Covariance matrix of asset returns for the four-asset portfolio over
         a one-year period.
        - `r_hat_1y_four`: Expected returns of the four assets over a one-year period.

    Returns:
    --------
    None
        This function does not return any values. It directly generates a plot comparing the
        efficient frontiers for the two portfolios and saves the plot as an image file.
    """
    compare_efficient_frontiers(cov_matrix_1y_five, r_hat_1y_five, cov_matrix_1y_four,
                                r_hat_1y_four, "1y")

def task_d_compare_2y():
    """
    Analogous to tne case of one year
    """
    compare_efficient_frontiers(cov_matrix_2y_five, r_hat_2y_five, cov_matrix_2y_four,
                                r_hat_2y_four, "2y")
