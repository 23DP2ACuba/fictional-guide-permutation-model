# Permutation model

## Features
OHLC Permutation: Generate statistically similar but time-shuffled market data while preserving:

Open-to-Close relationships

High/Low relative to Open

Log-normal price behavior

Strategy Evaluation: Compute approximate Sharpe ratio for strategy performance

Permutation Test: Assess whether a strategy's performance could occur by chance through comparison with permuted datasets

Visualization: Plot distribution of permuted strategy performances against real performance

## Functions
get_permutation(ohlc, start_index=0, seed=None)
Generates a permuted version of OHLC data while preserving key statistical properties.

### Parameters:

ohlc: Pandas DataFrame or list of DataFrames with OHLC data

start_index: Index from which to begin permutations (preserves initial data)

seed: Random seed for reproducibility

### Returns:

Permuted OHLC data with same structure as input

trading_strategy(ohlc)
Example moving average crossover strategy (9 vs 20 periods).

evaluate_strategy(ohlc, strategy_func)
Computes approximate Sharpe ratio of a strategy.

permutation_test(ohlc, strategy_func, n_perm=1000, seed=None)
Runs full permutation test workflow:

1. Evaluates strategy on real data

2. Generates n_perm permuted datasets

3. Evaluates strategy on each permuted dataset

4. Computes p-value

