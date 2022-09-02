import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns

tickers = ['HD', 'DIS','MSFT', 'BA', 'MMM', 'PFE', 'NKE', 'JNJ', 'MCD', 'INTC', 'XOM', 'GS', 'JPM', 'AXP', 'V', 'IBM', 'UNH', 'PG', 'GE', 'KO', 'CSCO', 'CVX', 'CAT', 'MRK', 'WMT', 'VZ', 'RTX', 'TRV', 'AAPL', 'ADBE', 'EBAY', 'QCOM', 'HPQ', 'JNPR', 'AMD']

def download_stocks_data():
    
    pd.core.common.is_list_like = pd.api.types.is_list_like
    from pandas_datareader import data as pdr
    import datetime
    import fix_yahoo_finance as yf
    
    start = datetime.datetime(2013, 1, 1)
    end = datetime.datetime(2018, 1, 1) 
    
    df = pdr.get_data_yahoo(tickers, start, end)['Close']
    
    df.to_csv('pairs_data.csv')
    

data = pd.read_csv('pairs_data.csv')
data.set_index('Date', inplace = True)


# Finding pairs of companies for which stock price movements are cointegrated
 
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
                
    return score_matrix, pvalue_matrix, pairs
    

# Heatmap to show p-value of cointegration between pairs of stocks

def heatmap(data):
    scores, pvalues, pairs = find_cointegrated_pairs(data)
    import seaborn
    fig, ax = plt.subplots(figsize = (10, 10))
    seaborn.heatmap(pvalues, xticklabels = tickers, yticklabels = tickers, cmap = 'RdYlGn_r', mask = (pvalues >= 0.01) )
    print (pairs)

# Lets select from the pairs to create a trading strategy based on those stocks

S1 = data['HD'].copy()   
S2 = data['TRV'].copy()    

score, pvalue, _ = coint(S1, S2)

# Calculating the Spread

def spread(S1, S2):
    
    
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    S1 = data['HD'].copy()
    b = results.params[1]
    
    spread = S2 - b * S1
    spread.plot(figsize=(12,6))
    plt.axhline(spread.mean(), color='black')
    plt.legend(['Spread']);
    
    # We can clearly see that the spread plot moves around mean


def ratio(S1, S2):
    ratio = S1 / S2
    ratio.plot(figsize = (12, 6))
    a = ratio.mean()
    plt.axhline(a, color = 'black')
    plt.legend(['Price Ratio'])
 
    # We now need to standardize this ratio (using z-score) because the absolute ratio might not be the most ideal way of analyzing this trend.

def zscore(S1, S2):
    ratio = S1 / S2
    b = ratio.mean()
    score =  (ratio - b) / np.std(ratio)
    a = score.mean()
    score.plot(figsize = (12, 6))
    plt.axhline(a)
    plt.axhline(1.0, color='red')
    plt.axhline(-1.0, color='green')
    plt.show()
    
ratios = S1 / S2
a = int (len(ratios) * 0.70)
train = ratios[:a]
test = ratios[a:]
    
def rolling_ratio_zscore(S1, S2):
    
    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    plt.figure(figsize=(12, 6))
    train.plot()
    ratios_mavg5.plot()
    ratios_mavg60.plot()
    plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])
    plt.ylabel('Ratio')
    plt.show()

def trade_signals(S1, S2):
    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    
    plt.figure(figsize=(12,6))
    zscore_60_5.plot()
    plt.axhline(0, color='black')
    plt.axhline(1.0, color='red', linestyle='--')
    plt.axhline(-1.0, color='green', linestyle='--')
    plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
    plt.show()



def ratio_with_signal(S1, S2):
    ratios_mavg5 = train.rolling(window=5, center=False).mean()
    ratios_mavg60 = train.rolling(window=60, center=False).mean()
    std_60 = train.rolling(window=60, center=False).std()
    zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
    
    plt.figure(figsize=(12,6))

    train[160:].plot()
    buy = train.copy()
    sell = train.copy()
    buy[zscore_60_5>-1] = 0
    sell[zscore_60_5<1] = 0
    buy[160:].plot(color='g', linestyle='None', marker='^')
    sell[160:].plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, ratios.min(), ratios.max()))
    plt.legend(['Ratio', 'Buy Signal', 'Sell Signal'])
    plt.show()

# Trading using a simple strategy

def trade(S1, S2, window1, window2):
    
    if (window1 == 0) or (window2 == 0):
        return 0
    
    ratios = S1/S2
    ma1 = ratios.rolling(window=window1,
                               center=False).mean()
    ma2 = ratios.rolling(window=window2,
                               center=False).mean()
    std = ratios.rolling(window=window2,
                        center=False).std()
    zscore = (ma1 - ma2)/std
    
    money = 1000
    countS1 = 0
    countS2 = 0
    max_positions = 5
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] > 1:
            if countS1 >= 0 and countS2 >= 0:
                money += S1[i] - S2[i] * ratios[i]
                countS1 -= 1
                countS2 += ratios[i]
                print('Selling Ratio %s %s %s %s'%(money, ratios[i], countS1,countS2))
        # Buy long if the z-score is < -1
        elif zscore[i] < -1:
            if countS1 <= max_positions and countS2 <= max_positions :
                money -= S1[i] - S2[i] * ratios[i]
                countS1 += 1
                countS2 -= ratios[i]
                print('Buying Ratio %s %s %s %s'%(money,ratios[i], countS1,countS2))
        # Clear positions if the z-score between -.75 and .75
        elif abs(zscore[i]) < 0.75:
            money += S1[i] * countS1 + S2[i] * countS2
            countS1 = 0
            countS2 = 0
            print('Exit pos %s %s %s %s'%(money,ratios[i], countS1,countS2))
    print (money)

trade(data.HD[a:], data.TRV[a:], 5, 60)
# ratio_with_signal(S1, S2)
# rolling_ratio_zscore(S1, S2)
# trade_signals(S1, S2)
# rolling_ratio_zscore(S1, S2)
# zscore(S1, S2)
# ratio(S1, S2)
# spread(S1, S2)
# heatmap(data)
# print (find_cointegrated_pairs(data))