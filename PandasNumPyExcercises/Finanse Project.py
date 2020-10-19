# imports
from pandas_datareader import data, wb
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf

cf.go_offline()

sns.set_style('whitegrid')
start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)
# stock information for the following banks:
# Bank of America
BAC = data.DataReader("BAC", 'stooq', start, end)

# CitiGroup
C = data.DataReader("C", 'stooq', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'stooq', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'stooq', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'stooq', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'stooq', start, end)

print(WFC.head())
# list of the ticker symbols
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC], axis=1, keys=tickers)
bank_stocks.columns.names = ['Bank Ticker','Stock Info']
print(bank_stocks.head())
# max Close price for each bank's stock throughout the time period
bank_stocks.xs(key='Close', axis=1, level='Stock Info').max()
returns = pd.DataFrame()
for tick in tickers:
    returns[tick + ' Return'] = bank_stocks[tick]['Close'].pct_change()
print(returns.head())
sns.pairplot(data=returns)
plt.show()
print(returns.idxmin())
# distplot using seaborn of the 2015 returns for Morgan Stanley
sns.distplot(returns.loc['2015-01-01':'2015-12-31']['MS Return'], bins=50)
plt.show()
sns.distplot(returns.loc['2008-01-01':'2008-12-31']['C Return'], bins=50)
plt.show()
bank_stocks.xs(key='Close', axis=1, level='Stock Info').plot()
plt.figure(figsize=(12, 6))
plt.show()
BAC['Close'].loc['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
tc = bank_stocks.xs(key='Close', axis=1, level='Stock Info').corr()
sns.heatmap(tc, annot=True, cmap='coolwarm')
plt.show()
