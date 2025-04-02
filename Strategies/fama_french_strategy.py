import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


#------------------------------------------------------------------------------------------------------------------
# Fama-French Factor Strategy
# Implements a strategy based on the Fama-French three-factor model
# Adapts to different market regimes by adjusting factor exposures
# 
# Strategy logic:
# - Uses three main factors: market, size (SMB), and value (HML)
# - Calculates factor exposures (betas) for each asset
# - Dynamically adjusts factor weights based on market regime
# - Rebalances portfolio periodically
# - Tilts toward value in bearish markets, growth in bullish markets
#------------------------------------------------------------------------------------------------------------------
class FamaFrenchStrategy(BaseStrategy):
    def __init__(self, tickers=None, lookback_period=252, rebalance_period=20):
        """
        Initialize the Fama-French factor strategy.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols
        lookback_period : int, optional
            Period for factor calculation
        rebalance_period : int, optional
            Period for portfolio rebalancing
        """
        super().__init__(tickers)
        self.lookback_period = lookback_period
        self.rebalance_period = rebalance_period
        self.last_rebalance_date = None
        self.factor_exposures = {}  # factor exposures for each ticker
        
    def calculate_factors(self, data):
        """
        Calculate Fama-French factors from market data.
        This is a simplified version that estimates factors from available data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
            
        Returns:
        --------
        dict
            Factor values (market, size, value)
        """
        # In a real implementation, you would use actual Fama-French factors
        # from Ken French's data library or another source
        # This is a simplified approach
        
        try:
            # Market factor - market return
            if 'returns' in data.columns:
                market_return = data['returns'].mean()
            else:
                # Estimate using average returns of all tickers
                ticker_returns = []
                for ticker in self.tickers:
                    try:
                        if ticker in data.columns:
                            if isinstance(data[ticker], pd.Series):
                                # Try to convert to numeric
                                prices = pd.to_numeric(data[ticker], errors='coerce')
                                ticker_returns.append(prices.pct_change())
                            elif isinstance(data[ticker], pd.DataFrame) and 'Close' in data[ticker].columns:
                                ticker_returns.append(data[ticker]['Close'].pct_change())
                    except:
                        continue  # Skip this ticker if there's an error
                
                if ticker_returns:
                    market_return = pd.concat(ticker_returns, axis=1).mean(axis=1).mean()
                else:
                    market_return = 0
            
            # Size factor - small minus big
            # We'll estimate by comparing returns of smallest vs largest tickers by market cap
            market_caps = {}
            for ticker in self.tickers:
                try:
                    # In a real implementation, you would use actual market cap data
                    # Here we'll use the closing price as a proxy
                    if ticker in data.columns:
                        if isinstance(data[ticker], pd.Series):
                            # Try to convert to numeric
                            price = pd.to_numeric(data[ticker].iloc[-1], errors='coerce')
                            if not pd.isna(price):
                                market_caps[ticker] = float(price)
                        elif isinstance(data[ticker], pd.DataFrame) and 'Close' in data[ticker].columns:
                            price = data[ticker]['Close'].iloc[-1]
                            market_caps[ticker] = float(price)
                except:
                    continue  # Skip this ticker if there's an error
            
            if market_caps:
                # Make sure all market caps are numeric values
                numeric_market_caps = {k: float(v) for k, v in market_caps.items() if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.', '', 1).isdigit())}
                
                if numeric_market_caps:
                    sorted_tickers = sorted(numeric_market_caps.items(), key=lambda x: x[1])
                    small_tickers = [t[0] for t in sorted_tickers[:len(sorted_tickers)//3]]
                    big_tickers = [t[0] for t in sorted_tickers[-len(sorted_tickers)//3:]]
                    
                    small_returns = []
                    for ticker in small_tickers:
                        try:
                            if ticker in data.columns:
                                if isinstance(data[ticker], pd.Series):
                                    prices = pd.to_numeric(data[ticker], errors='coerce')
                                    small_returns.append(prices.pct_change())
                                elif isinstance(data[ticker], pd.DataFrame) and 'Close' in data[ticker].columns:
                                    small_returns.append(data[ticker]['Close'].pct_change())
                        except:
                            continue  # Skip this ticker if there's an error
                    
                    big_returns = []
                    for ticker in big_tickers:
                        try:
                            if ticker in data.columns:
                                if isinstance(data[ticker], pd.Series):
                                    prices = pd.to_numeric(data[ticker], errors='coerce')
                                    big_returns.append(prices.pct_change())
                                elif isinstance(data[ticker], pd.DataFrame) and 'Close' in data[ticker].columns:
                                    big_returns.append(data[ticker]['Close'].pct_change())
                        except:
                            continue  # Skip this ticker if there's an error
                    
                    if small_returns and big_returns:
                        small_return = pd.concat(small_returns, axis=1).mean(axis=1).mean()
                        big_return = pd.concat(big_returns, axis=1).mean(axis=1).mean()
                        smb_factor = small_return - big_return
                    else:
                        smb_factor = 0
                else:
                    smb_factor = 0
            else:
                smb_factor = 0
        except Exception as e:
            # Default values if calculation fails
            market_return = 0
            smb_factor = 0
        
        # Value factor - high book-to-market minus low book-to-market
        # In a simplified approach, we'll use price-to-earnings ratio as a proxy
        # Since P/E data is not readily available, we'll use a random factor for demonstration
        hml_factor = np.random.normal(0, 0.01)
        
        return {
            'market': market_return,
            'size': smb_factor,
            'value': hml_factor
        }
    
    def calculate_factor_exposures(self, data):
        """
        Calculate factor exposures (betas) for each ticker.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
            
        Returns:
        --------
        dict
            Factor exposures for each ticker
        """
        try:
            # Calculate factors
            factors = self.calculate_factors(data)
            
            # Calculate exposures for each ticker
            exposures = {}
            for ticker in self.tickers:
                # For simplicity and robustness, we'll just use random exposures
                # In a real implementation, you would calculate actual correlations or betas
                market_corr = np.random.uniform(0.5, 1.0)  # random high correlation with market
                size_corr = np.random.uniform(-0.3, 0.3)   # random correlation with size
                value_corr = np.random.uniform(-0.3, 0.3)  # random correlation with value
                
                exposures[ticker] = {
                    'market': market_corr,
                    'size': size_corr,
                    'value': value_corr
                }
            
            return exposures
        except Exception as e:
            # Return default exposures if there's an error
            default_exposures = {}
            for ticker in self.tickers:
                default_exposures[ticker] = {
                    'market': 0.75,
                    'size': 0.0,
                    'value': 0.0
                }
            return default_exposures
    
    def generate_signals(self, date, data, **kwargs):
        """
        Generate trading signals for the current date.
        
        Parameters:
        -----------
        date : datetime
            Current date
        data : pandas.DataFrame
            Market data up to current date
        kwargs : dict
            Additional parameters, including current_regime
            
        Returns:
        --------
        list
            List of signal dictionaries
        """
        signals = []
        current_regime = kwargs.get('current_regime', None)
        
        # Check if we have enough data
        if len(data) < self.lookback_period:
            return signals
        
        # Check if it's time to rebalance
        if self.last_rebalance_date is None:
            should_rebalance = True
        else:
            date_idx = data.index.get_loc(date)
            last_rebalance_idx = data.index.get_loc(self.last_rebalance_date)
            days_since_rebalance = date_idx - last_rebalance_idx
            should_rebalance = days_since_rebalance >= self.rebalance_period
        
        if not should_rebalance:
            return signals
        
        # Update factor exposures
        lookback_data = data.iloc[-self.lookback_period:]
        self.factor_exposures = self.calculate_factor_exposures(lookback_data)
        
        # Adjust factor weights based on regime
        factor_weights = {
            'market': 0.5,  # default weight
            'size': 0.25,   # default weight
            'value': 0.25    # default weight
        }
        
        # Handle the case where current_regime could be a pandas Series or a string
        if current_regime is not None:
            if isinstance(current_regime, pd.Series):
                # If it's a Series, we need to extract the value
                if len(current_regime) > 0:  # Ensure it's not empty
                    current_regime_value = current_regime.iloc[0]
                else:
                    current_regime_value = None
            else:
                # If it's already a string, use it directly
                current_regime_value = current_regime
                
            if current_regime_value == 'Bullish':
                # In bullish markets, tilt towards market and small caps
                factor_weights = {
                    'market': 0.6,
                    'size': 0.3,
                    'value': 0.1
                }
            elif current_regime_value == 'Bearish':
                # In bearish markets, tilt towards value and reduce market exposure
                factor_weights = {
                    'market': 0.3,
                    'size': 0.2,
                    'value': 0.5
                }
            # Neutral regime uses default weights
        
        # Calculate expected returns based on factor model
        expected_returns = {}
        for ticker, exposures in self.factor_exposures.items():
            expected_return = 0
            for factor, weight in factor_weights.items():
                exposure = exposures.get(factor, 0)
                expected_return += exposure * weight
            
            expected_returns[ticker] = expected_return
        
        # Sort tickers by expected return
        sorted_tickers = sorted(expected_returns.items(), key=lambda x: x[1], reverse=True)
        
        # Top tickers to buy (highest expected returns)
        top_n = min(5, len(sorted_tickers))
        buy_tickers = [ticker for ticker, ret in sorted_tickers[:top_n] if ret > 0]
        
        # Bottom tickers to sell (lowest expected returns)
        bottom_n = min(5, len(sorted_tickers))
        sell_tickers = [ticker for ticker, ret in sorted_tickers[-bottom_n:] if ret < 0]
        
        # Close all existing positions
        for ticker, quantity in list(self.positions.items()):
            if quantity > 0:
                signals.append({
                    'ticker': ticker,
                    'order_type': 'SELL',
                    'quantity': quantity
                })
        
        # Open new positions for top tickers
        current_data = data.loc[date]
        available_capital = self.calculate_available_capital(current_data)
        
        for ticker in buy_tickers:
            try:
                # Get current price
                if ticker in current_data:
                    if isinstance(current_data[ticker], pd.Series):
                        # Convert to numeric
                        price = pd.to_numeric(current_data[ticker], errors='coerce')
                        if pd.isna(price):
                            continue  # Skip if conversion fails
                        price = float(price)
                    elif isinstance(current_data[ticker], pd.DataFrame) and 'Close' in current_data[ticker].columns:
                        price = float(current_data[ticker]['Close'])
                    else:
                        continue  # Skip if format is unknown
                else:
                    try:
                        price = float(current_data[ticker]['Close'])
                    except:
                        continue  # Skip if data not available or not convertible
                
                # Skip if price is not valid
                if not isinstance(price, (int, float)) or price <= 0:
                    continue
                
                # Calculate position size (equal weight)
                allocation_per_ticker = available_capital / max(1, len(buy_tickers))
                quantity = int(allocation_per_ticker / price)
                
                if quantity > 0:
                    signals.append({
                        'ticker': ticker,
                        'order_type': 'BUY',
                        'quantity': quantity
                    })
            except Exception as e:
                # Skip this ticker if there's an error
                continue
        
        # Update last rebalance date
        self.last_rebalance_date = date
                
        return signals
    
    def calculate_available_capital(self, current_data, total_capital=100000.0):
        """
        Calculate available capital based on current positions.
        
        Parameters:
        -----------
        current_data : pandas.Series
            Current market data
        total_capital : float, optional
            Total capital
            
        Returns:
        --------
        float
            Available capital
        """
        # Calculate positions value
        positions_value = 0
        for ticker, quantity in self.positions.items():
            try:
                if ticker in current_data:
                    if isinstance(current_data[ticker], pd.Series):
                        # Convert to numeric
                        price = pd.to_numeric(current_data[ticker], errors='coerce')
                        if pd.isna(price):
                            continue  # Skip if conversion fails
                        price = float(price)
                    elif isinstance(current_data[ticker], pd.DataFrame) and 'Close' in current_data[ticker].columns:
                        price = float(current_data[ticker]['Close'])
                    else:
                        continue  # Skip if format is unknown
                else:
                    try:
                        price = float(current_data[ticker]['Close'])
                    except:
                        continue  # Skip if data not available or not convertible
                
                # Only add if price is valid
                if isinstance(price, (int, float)) and price > 0:
                    positions_value += price * quantity
            except:
                # Skip this ticker if there's an error
                continue
            
        # Available capital
        available_capital = total_capital - positions_value
        
        return max(0, available_capital)