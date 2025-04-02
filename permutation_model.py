import numpy as np
import pandas as pd
from typing import List, Union

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])
    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']])
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        r_o = (log_bars['Open'] - log_bars['Close'].shift()).to_numpy()
        r_h = (log_bars['High'] - log_bars['Open']).to_numpy()
        r_l = (log_bars['Low'] - log_bars['Open']).to_numpy()
        r_c = (log_bars['Close'] - log_bars['Open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))
        log_bars = np.log(reg_bars[['Open', 'High', 'Low', 'Close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(perm_bars, index=time_index, columns=['Open', 'High', 'Low', 'Close'])
        perm_ohlc.append(perm_bars)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]


def trading_strategy(ohlc: pd.DataFrame) -> pd.Series:
    """
    A simple moving average crossover strategy.
    Generates a long signal when the short-term moving average is above the long-term moving average.
    """
    sma_short = ohlc['Close'].rolling(window=9).mean()
    sma_long = ohlc['Close'].rolling(window=20).mean()
    
    signal = (sma_short > sma_long).astype(int)
    signal = signal.shift(1).fillna(0)
    
    returns = ohlc['Close'].pct_change() * signal
    return returns

def evaluate_strategy(ohlc: pd.DataFrame, strategy_func) -> float:
    """
    Evaluates the strategy performance by computing an approximate Sharpe ratio.
    """
    returns = strategy_func(ohlc)
    if returns.std() == 0:
        return 0.0
    sharpe_ratio = returns.mean() / returns.std()
    return sharpe_ratio

def permutation_test(ohlc: pd.DataFrame, strategy_func, n_perm=1000, seed=None):
    """
    Runs a permutation test:
      - Computes the real strategy performance on the original data.
      - Generates n_perm permuted datasets and evaluates performance on each.
      - Computes a p-value as the fraction of permuted performances that equal or exceed the real performance.
    """
    real_perf = evaluate_strategy(ohlc, strategy_func)
    perm_perf = []
    
    for i in range(n_perm):
        perm_data = get_permutation(ohlc, seed=seed+i if seed is not None else None)
        perf = evaluate_strategy(perm_data, strategy_func)
        perm_perf.append(perf)
    
    perm_perf = np.array(perm_perf)
    p_value = np.mean(perm_perf >= real_perf)
    return real_perf, perm_perf, p_value


if __name__ == '__main__':
    import yfinance
    import matplotlib.pyplot as plt
    
    data = yfinance.Ticker("TSLA").history(start="2020-01-01", end="2025-01-01")
    
    real_perf, perm_perf, p_value = permutation_test(data, trading_strategy, n_perm=500, seed=42)
    
    print(f"Real Strategy Sharpe Ratio: {real_perf:.4f}")
    print(f"P-value from Permutation Test: {p_value:.4f}")
    
    plt.hist(perm_perf, bins=30, alpha=0.7, label='Permuted Sharpe Ratios')
    plt.axvline(real_perf, color='red', linestyle='dashed', linewidth=2, label='Real Strategy Sharpe Ratio')
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Permutation Test for Trading Strategy Performance")
    plt.show()
