import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
from numpy import linalg as LA
import yfinance as yf

print("\n__________________ Stock Portfolio Simulator __________________\n\n")
number_of_stocks = int(input("How Much Stocks you Want on Portfolio?\n"))

stocks_list = []

for s in range(number_of_stocks):  
    stocks_list.append(input("Code (On Capital letters) of Stock " + str(len(stocks_list) + 1) + " \n")) 

stocks_list = [stock + '.SA' for stock in stocks_list]

final_date = dt.datetime.now()
initial_date = final_date - dt.timedelta(days=300)

print("Getting Stocks Info...")
prices = yf.download(stocks_list, initial_date, final_date)['Adj Close']

stock_returns = prices.pct_change().dropna()
covarience_matrix = stock_returns.cov()
portfolio_weight = np.full(len(stocks_list), 1/len(stocks_list))
number_of_stocks = len(stocks_list)

number_of_simulations = int(input("How Much Portfolio Simulations you Want to Do? \n"))
projected_days = 252 * int(input("For how much Years You want to Simulate Each Portfolio? \n"))
initial_capital  = int(input("What's your Initial capital? \n"))


avarage_returns = stock_returns.mean(axis=0).to_numpy()
matrix_avarage_returns = avarage_returns * np.ones(shape = (projected_days, number_of_stocks))

L = LA.cholesky(covarience_matrix)


portfolio_returns = np.zeros([projected_days, number_of_simulations])
final_amount = np.zeros(number_of_simulations)

for s in range(number_of_simulations): 

    Rpdf = np.random.normal(size=(projected_days, number_of_stocks))

    sintetic_returns = matrix_avarage_returns + np.inner(Rpdf, L)

    portfolio_returns[:, s] = np.cumprod(np.inner(portfolio_weight, sintetic_returns) + 1) * initial_capital

    final_amount[s] = portfolio_returns[-1, s]

amount_99 = str(np.percentile(final_amount, 1))
amount_95 = str(np.percentile(final_amount, 5))
avarage_amount = str(np.percentile(final_amount, 50))
profit_scenarios = str((len(final_amount[final_amount > initial_capital])/ 
len(final_amount)) * 100) + "%"

print("\n_____________ Final Results _____________")

print(f'''\n \n When Investing {initial_capital} on a portfolio with {stocks_list} by {projected_days / 252} 
        years, using Monte Carlo Algorith to generate {number_of_simulations}, we discovered:

    50% of probability that the final amount is higher than {avarage_amount},
    95% of probability that the final amount is higher than {amount_95} and
    95% of probability that the final amount is higher than {amount_99}.

    In {profit_scenarios[:2]} times, it could be possible to have profit on the next {projected_days / 252} years!
''')

plt.plot(portfolio_returns, linewidth=1)
plt.xlabel('Money')
plt.ylabel('Days')
plt.show()


