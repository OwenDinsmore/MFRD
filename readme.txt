|----Overview----------------------|
Multi-Factor Regime Detection (MFRD) System
============================================

MFRD is a Python-based trading system that detects market regimes (bullish, bearish, neutral) and applies appropriate trading strategies for each regime.

|----Components--------------------|
1. HMM - Hidden Markov Model for regime detection
2. Model - Core system integrating regime detection with trading strategies
3. Simulator - Backtesting engine for strategy evaluation
4. Strategies - Trading strategies adapted to different market regimes

|----Installation-------------------|
The system requires the following Python packages:
- numpy
- pandas
- matplotlib
- hmmlearn
- scikit-learn
- yfinance (for data fetching)

Install dependencies with:
```
pip install numpy pandas matplotlib hmmlearn scikit-learn yfinance
```

|----Usage--------------------------|
Run the main.py script to execute a demonstration of the system:
```
python main.py
```

This will:
1. Load historical market data
2. Detect market regimes using the HMM model
3. Backtest multiple trading strategies
4. Compare performance across different regimes
5. Generate performance visualization

|----Hidden Markov Model (HMM)-----|
Regime detection with Hidden Markov Model to find market states:
- Identifies three distinct market regimes (bullish, bearish, neutral)
- Uses return distributions and volatility patterns
- Calculates transition probabilities between states
- Assigns regime labels automatically based on return characteristics
- Provides confidence probabilities for each regime

|----Strategies--------------------|
The system includes the following strategies:

1. Momentum Strategy - Performs well in bullish markets
   - Buys assets with strong positive momentum
   - Exits positions in bearish regimes

2. Mean Reversion Strategy - Performs well in neutral/sideways markets
   - Buys oversold assets and sells overbought assets based on RSI
   - Adjusts position sizing based on market regime

3. Fama-French Strategy - Adapts factor exposures to market regimes
   - Tilts towards market and size factors in bullish regimes
   - Increases value factor exposure in bearish regimes

4. Regime Adaptive Strategy - Dynamically switches between strategies
   - Uses momentum in bullish markets
   - Uses mean reversion in neutral markets
   - Uses defensive factor allocation in bearish markets

|----Simulator---------------------|
The simulator component handles:
- Portfolio management and position tracking
- Order execution with realistic constraints
- Performance measurement and reporting
- Detailed transaction logging
- Risk-based position sizing

|----Customization-----------------|
To use the system with your own data and strategies:

1. Create a Model instance
2. Load data with model.load_data() or provide your own DataFrame
3. Register strategies with model.register_strategy()
4. Run backtests with model.backtest_all_strategies()
5. Analyze results with model.plot_strategy_performance()

Example:
```python
from Model import Model
from Strategies import MomentumStrategy

# Initialize model
model = Model()

# Load data
model.load_data("AAPL", "2020-01-01", "2023-01-01")

# Register strategies
model.register_strategy("Momentum", MomentumStrategy)

# Run backtest
results = model.backtest_all_strategies()

# Plot results
model.plot_strategy_performance(benchmark="SPY")
```

For more detailed documentation, refer to the docstrings in the source code.