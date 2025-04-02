import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from HMM.hmm import MarketRegimeHMM

print('MFRD System Live View Test')
print('========================')
print('Training HMM on SPY data, testing strategies on individual tickers')

# Simulate fetching data for SPY and individual tickers
def fetch_market_data(ticker, start_date, end_date, seed=42):
    '''Simulate fetching market data'''
    np.random.seed(seed + hash(ticker) % 100)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Set ticker-specific parameters
    if ticker == 'SPY':
        mean_return = 0.0005
        volatility = 0.012
        drift = 0.0001
    elif ticker == 'QQQ':
        mean_return = 0.0007
        volatility = 0.016
        drift = 0.0002
    else:
        mean_return = 0.0005
        volatility = 0.015
        drift = 0.0001
    
    # Create price series
    prices = [100.0]
    regime_change = n_days // 2
    
    for i in range(1, n_days):
        if i < regime_change:
            # Bull market
            day_drift = drift * 2
            day_vol = volatility * 0.8
        else:
            # Bear market
            day_drift = drift * -1
            day_vol = volatility * 1.5
        
        daily_return = np.random.normal(mean_return + day_drift, day_vol)
        prices.append(prices[-1] * (1 + daily_return))
    
    # Create DataFrame
    df = pd.DataFrame({
        'close': prices,
        'returns': [0] + [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
    }, index=dates)
    
    # Calculate volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Calculate moving averages
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Normalize features to -1 to 1 scale
    df['returns_norm'] = df['returns'].clip(-0.1, 0.1) / 0.1
    mean_vol = df['volatility'].mean()
    df['volatility_norm'] = df['volatility'] / (2 * mean_vol) - 0.5
    df['volatility_norm'] = df['volatility_norm'].clip(-1, 1)
    df['rsi_norm'] = (df['rsi'] - 50) / 50
    df['rsi_norm'] = df['rsi_norm'].clip(-1, 1)
    
    # Calculate price/MA ratios
    df['price_sma50_ratio'] = (df['close'] / df['sma_50'] - 1) * 5
    df['price_sma50_ratio'] = df['price_sma50_ratio'].clip(-1, 1)
    df['sma50_sma200_ratio'] = (df['sma_50'] / df['sma_200'] - 1) * 5
    df['sma50_sma200_ratio'] = df['sma50_sma200_ratio'].clip(-1, 1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

# Calculate strategy weights
def calculate_strategy_weights(regime_score, features):
    '''Calculate strategy weights based on regime and features'''
    weights = {
        'Momentum': 0.3 + 0.7 * regime_score,
        'Mean Reversion': 0.4 - 0.5 * regime_score,
        'Value': 0.2 - 0.1 * abs(regime_score),
        'Growth': 0.1 + 0.4 * regime_score
    }
    
    # Adjust based on other features
    if features.get('volatility_norm', 0) > 0.5:
        weights['Mean Reversion'] += 0.2
        weights['Momentum'] -= 0.1
    
    if features.get('rsi_norm', 0) > 0.7:
        weights['Momentum'] -= 0.2
        weights['Mean Reversion'] += 0.2
    elif features.get('rsi_norm', 0) < -0.7:
        weights['Mean Reversion'] -= 0.2
        weights['Momentum'] += 0.2
    
    # Clip to -1 to 1 range
    for strat in weights:
        weights[strat] = max(-1, min(1, weights[strat]))
    
    return weights

# Calculate allocations
def calculate_allocation(weights, cash=100000):
    '''Convert weights to dollar allocations'''
    allocation = {}
    
    # Separate long and short weights
    longs = {k: v for k, v in weights.items() if v > 0}
    shorts = {k: v for k, v in weights.items() if v < 0}
    
    # Calculate allocation percentages
    total_long = sum(longs.values())
    total_short = sum(abs(v) for v in shorts.values())
    
    # Long positions
    long_cash = cash * 0.9
    if total_long > 0:
        for strategy, weight in longs.items():
            allocation[strategy] = (weight / total_long) * long_cash
    
    # Short positions
    short_cash = cash * 0.5
    if total_short > 0:
        for strategy, weight in shorts.items():
            allocation[strategy] = (weight / total_short) * short_cash
    
    return allocation

# Print allocation table
def print_allocation_table(allocations, weights):
    '''Print a formatted table of allocations'''
    print('┌─────────────────┬───────────┬──────────────┬────────────────┐')
    print('│ Strategy        │ Direction │ Signal (-1,1)│ Allocation ($) │')
    print('├─────────────────┼───────────┼──────────────┼────────────────┤')
    for strategy, amount in allocations.items():
        direction = 'LONG' if amount > 0 else 'SHORT'
        signal = weights[strategy]
        print(f'│ {strategy:<15} │ {direction:<9} │ {signal:+6.2f}      │ ${abs(amount):<14,.2f} │')
    print('└─────────────────┴───────────┴──────────────┴────────────────┘')

# Simulate the MFRD system
def run_live_simulation():
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    print(f'Simulation Period: {start_date} to {end_date}')
    
    # Step 1: Get market data
    print('Fetching market data...')
    spy_data = fetch_market_data('SPY', start_date, end_date, seed=42)
    tickers = ['QQQ', 'IWM']
    ticker_data = {ticker: fetch_market_data(ticker, start_date, end_date, seed=42) for ticker in tickers}
    
    # Step 2: Train HMM on SPY
    print('Training HMM model on SPY data...')
    hmm_model = MarketRegimeHMM(n_states=3, n_iter=100, random_state=42)
    hmm_model.fit(spy_data)
    
    # Step 3: Simulate live trading for last 30 days
    print('Simulating live trading for last 30 days...')
    
    # Get common dates
    common_dates = spy_data.index
    for ticker, data in ticker_data.items():
        common_dates = common_dates.intersection(data.index)
    
    # Filter to common dates
    spy_data = spy_data.loc[common_dates]
    for ticker in tickers:
        ticker_data[ticker] = ticker_data[ticker].loc[common_dates]
    
    # Use last 30 days
    simulation_dates = common_dates[-30:]
    spy_states = hmm_model.states[-30:]
    
    # Print regime distribution
    print('Regime distribution during simulation:')
    state_counts = pd.Series(spy_states).value_counts()
    for state, count in state_counts.items():
        label = hmm_model.state_labels[state]
        score = hmm_model.state_indices[state]
        pct = count / len(spy_states) * 100
        print(f' - State {state} ({label}, score {score}): {count} days ({pct:.1f}%)')
    
    # Get latest date results
    last_date = simulation_dates[-1]
    last_state = spy_states[-1]
    last_regime = hmm_model.state_labels[last_state]
    last_score = hmm_model.state_indices[last_state]
    
    # Get features for latest date
    spy_features = spy_data.loc[last_date]
    ticker_features = {ticker: data.loc[last_date] for ticker, data in ticker_data.items()}
    
    # Combine features
    combined_features = {
        'regime_score': last_score,
        'returns_norm': spy_features['returns_norm'],
        'volatility_norm': spy_features['volatility_norm'],
        'rsi_norm': spy_features['rsi_norm'],
        'price_sma50_ratio': spy_features['price_sma50_ratio'],
        'sma50_sma200_ratio': spy_features['sma50_sma200_ratio']
    }
    
    # Add features from individual tickers
    for ticker, features in ticker_features.items():
        combined_features[f'{ticker}_returns'] = features['returns_norm']
        combined_features[f'{ticker}_rsi'] = features['rsi_norm']
        combined_features[f'{ticker}_trend'] = features['price_sma50_ratio']
    
    # Calculate strategy weights and allocations
    weights = calculate_strategy_weights(last_score, combined_features)
    allocations = calculate_allocation(weights)
    
    # Print results
    print('Live Simulation Results:')
    print(f'Current date: {last_date.strftime("%Y-%m-%d")}')
    print(f'Current market regime: {last_regime} (score: {last_score})')
    print(f'SPY closing price: ${spy_features["close"]:.2f}')
    
    print('Current Strategy Allocation:')
    print_allocation_table(allocations, weights)
    print('Live simulation completed successfully!')

if __name__ == "__main__":
    run_live_simulation()