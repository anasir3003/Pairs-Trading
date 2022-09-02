# Pairs-Trading
Implemented Pairs Trading Strategy after selecting pair from a pool of 35 high volume U.S. stocks in period of 2013-2018


Pairs trading is a form of mean-reversion that has a distinct advantage of always being hedged against market movements. The motivation for trading pairs has its roots in works that preach the existence of long term relation between stocks. If there exists indeed a long term equilibrium, deviation from this relation are expected to revert.Its simplest statistical arbitrage strategy as it contains a portfoilio of only two stocks. Two most popular methods for selecting pairs are based on cointegration(which I used to implement here) and distance method.

First I have downloaded OHLC data for 35 high volume U.S. stocks. After that I have written a function to find the highly cointegrated pairs with p-values < 0.05 using the Engle-Granger Cointegration test. Then using the heatmap of p-values I found the pair of stocks HD and TRV to have a really low p-value and hence implemented further trading on this pair. Firstly I found the spread, ratio for this pair. Then I generaed signals based on the z-score of ratios. Then I split the time series into train and test data and used following method to trade :

1 - If z-score < -1 you buy the ratio (as you expect it to revert back), i.e. you buy stock S1 and sell S2

2 - If z-score > 1 you sell the ratio (as you expect it to revert back), i.e. you sell stock S1 and buy S2

3 - If |z-score| < 0.75, you close the position in S1 and S2

This strategy generated return of 11.91278%
