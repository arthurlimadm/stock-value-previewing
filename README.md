# Stock Value Previewing with Python

## Abstract

This project will basically get a stock portofolio composed by 5 stock (on IBOVESPA market) and generate a preview of the portfolio’s future performance  by using as parameters it’s past performance, the preview generating is going to be generated by using Monte Carlo Alghoritm.

## What is Monte Carlo Alghoritm?

In [computing](https://en.wikipedia.org/wiki/Computing), a **Monte Carlo algorithm** is a [randomized algorithm](https://en.wikipedia.org/wiki/Randomized_algorithm) whose output may be incorrect with a certain (typically small) [probability](https://en.wikipedia.org/wiki/Probability). Two examples of such algorithms are [Karger–Stein algorithm](https://en.wikipedia.org/wiki/Karger%27s_algorithm)[[1]](https://en.wikipedia.org/wiki/Monte_Carlo_algorithm#cite_note-1) and Monte Carlo algorithm for [minimum Feedback arc set](https://en.wikipedia.org/wiki/Minimum_feedback_arc_set).[[2]](https://en.wikipedia.org/wiki/Monte_Carlo_algorithm#cite_note-2)

The name refers to the grand [casino](https://en.wikipedia.org/wiki/Monte_Carlo_Casino) in the Principality of Monaco at [Monte Carlo](https://en.wikipedia.org/wiki/Monte_Carlo), which is well-known around the world as an icon of gambling. The term "Monte Carlo" was first introduced in 1947 by [Nicholas Metropolis](https://en.wikipedia.org/wiki/Nicholas_Metropolis).[[3]](https://en.wikipedia.org/wiki/Monte_Carlo_algorithm#cite_note-3)

[Las Vegas algorithms](https://en.wikipedia.org/wiki/Las_Vegas_algorithm) are a [dual](https://en.wikipedia.org/wiki/Dual_(mathematics)) of Monte Carlo algorithms that never return an incorrect answer. However, they may make random choices as part of their work. As a result, the time taken might vary between runs, even with the same input.

If there is a procedure for verifying whether the answer given by a Monte Carlo algorithm is correct, and the probability of a correct answer is bounded above zero, then with probability, one running the algorithm repeatedly while testing the answers will eventually give a correct answer. Whether this process is a Las Vegas algorithm depends on whether halting with probability one is considered to satisfy the definition.

## Installing dependencies

To start our project, we are going to use some Packages, so install them:

```bash
pip install pandas_datareader numpy matplotlib datetime yfinance --user 
```

Now we can already start our application

## Catching Data

First we are going to define and array of stocks that will compose our **stock portfolio**;

```python
# importing dependencies
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
from numpy import linalg as LA
import yfinance as yf

# getting stocks data
stocks_list = ['WEGE3', 'PETR4', 'MGLU3', 'AMER3', 'ITSA4']

# converting into formated data
stocks_list = [stock + '.SA' for stock in stocks_list]
```

Note that we must format the data by adding an ‘SA’ to every stock string, we have to do this because we are gonna use yahoo finance to cathc financial data of the stocks, and yahoo finance api only accepts this specific format of data.

## Getting initial and final data;

We are going to use datetime to manipualte date and set a initial and final date:

```python
final_date = dt.datetime.now()
initial_date = final_date - dt.timedelta(days=300)
```

final_date if gonna be today and initial_date is gonna be 300 days ago.

This way we can acces the historic of past 300 years of price for ours stocks.

## Getting financial info of the stocks

To get stock’s prices, we are gonna use yahoo finance by pandas datareader.

Pandas datareader already has most of financial APIs to get financial data;

We are goingo to pass the array of stocks and the dates as an arguments for the function pdr.get_data_yahoo.

```python
prices = yf.download(stocks_list, initial_date, final_date)['Adj Close']
```

‘Ajd Close’ specify that we are going to store the adjusted market closing data.

## Calculating Stocks Preformance

To preview data we have to know wich is the Stocks return along the past time, to get to this value we can use **Covariance Matrix by extracting 5 values from the prices;**

### Code to get the Data

```python
stock_returns = prices.pct_change().dropna()
average_stock_returns = stock_returns.mean()
covarience_matrix = stock_returns.cov()
portfolio_weight = np.full(len(stocks_list), 1/len(stocks_list))
number_of_stocks = len(stocks_list)
```

### What each data represents?

1. Stock Returns
    1. Stock returns along the time (percentage of gain)
    
    
2. Avarage Stock Returns
    1. The mean of all days return belong the entire 300 days.
   
    
3. Covariance Matrix
    1. In [probability theory](https://en.wikipedia.org/wiki/Probability_theory) and [statistics](https://en.wikipedia.org/wiki/Statistics), a **covariance matrix** (also known as **auto-covariance matrix**, **dispersion matrix**, **variance matrix**, or **variance–covariance matrix**) is a square [matrix](https://en.wikipedia.org/wiki/Matrix_(mathematics)) giving the [covariance](https://en.wikipedia.org/wiki/Covariance) between each pair of elements of a given [random vector](https://en.wikipedia.org/wiki/Random_vector). Any [covariance](https://en.wikipedia.org/wiki/Covariance) matrix is [symmetric](https://en.wikipedia.org/wiki/Symmetric_matrix) and [positive semi-definite](https://en.wikipedia.org/wiki/Positive_semi-definite_matrix) and its main diagonal contains [variances](https://en.wikipedia.org/wiki/Variance) (i.e., the covariance of each element with itself).Intuitively, the covariance matrix generalizes the notion of variance to multiple dimensions. As an example, the variation in a collection of random points in two-dimensional space cannot be characterized fully by a single number, nor would the variances in thex and ydirections contain all of the necessary information; a 2×2 matrix would be necessary to fully characterize the two-dimensional variation.
   
    
4. Portfolio Weight
    1. Percentage of each stock on the entire portfolio.
    2. In this case (1 at 5) = 20%
5. Number of stocks
    1. The quantity of stocks on the portfolio.
    

## Simulating Hipotesys

We are gonna simulate 10 thousand Hipotesys of performance of our portfolio based on the past results and get the mean of them

To calculate Sintetic Returns, uses this formula:

$$
Sintetic Returns = Avarage Stock returns + Rpdf
$$

Where:

> Rpdf = Random Matrix generated by a probability funciton.
> 
> 
> Implements randobility to the result.
> 

Basically, on the stock market, the stocks have correlative relations, as an exmaple, if all market is rising, the probability of a stock to rise is higher.

But there is a problem, becuase on generating random probabilities on Python, we can’t ignore this.

So we are gonna use the covariant matrix to increment this by a special ‘L’ value on our formula;

$$
Sintetic Returns = Avarags Stock Returns + Rpdf . L
$$

Where:

> L = Triangular Matrix got from Choensky, using as base the past’s data covariance matrix.
> 
> 
> Uses past data as basis
> 

## Monte Carlo Premissing

So let’s start our Monte Carlo configuring by defining how much simulation we are going to generate, how much time we are going to project and the initial capital;

```python
# Starting with onw thousando simulations
simulations_number = 1000

# Here we are supposing a year have 252 days
projected_days = 252 * 3

# Budjet of 1000 dolars for our portfolio
initial_capital = 1000
```

And then we define the mean of this simulations:

```python
avarage_returns = stock_returns.mean(axis=0).to_numpy()
```

As we are summing the avarage_returns to the Rpdf, wich is a matrix, we must also convert the avarage_returns value into a matrix too:

```python
matrix_avarage_returns = avarage_returns * np.ones(shape = (projected_days, number_of_stocks))
```

## Generating L value

To generate L value we are gonna use a Linear Algebra module import as LA from the numpy, collet the Cholensky value and pass our covariance matrix as argument

```python
L = LA.cholensky(covariance_matrix)
```

## Generating Simulations

So we are going to use a for loop to generate the simulations;

```python
portfolio_returns = np.zeros([projected_days, number_of_simulations])
final_amount = np.zeros(number_of_simulations)

for s in range(number_of_simulations): 

		# Generating random Rpdf Matrix
    Rpdf = np.random.normal(size=(projected_days, number_of_stocks))

		# Sintetic Retuns Formula
    sintetic_returns = matrix_avarage_returns + np.inner(Rpdf, L)

		portfolio_returns[:, s] = np.cumprod(np.inner(portfolio_weight, sintetic_returns) + 1) * initial_capital

    final_amount[s] = portfolio_returns[-1, s]
```

## Plotting Chart

We are using matplotlib.pyplot.plot method to plot the chart:

```python
plt.plot(portfolio_returns, linewidth=1)
plt.xlabel('Money')
plt.ylabel('Days')
plt.show()
```
