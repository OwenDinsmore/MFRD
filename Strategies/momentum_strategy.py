import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


#------------------------------------------------------------------------------------------------------------------
# Momentum Strategy
# Uses price momentum to generate trading signals
# Performs well in strong bull markets
# 
# Strategy logic:
# - Buy assets with strongest price momentum (best recent performance)
# - Hold for a specified period or until momentum weakens
# - Exit all positions when market regime turns bearish
# - Particularly effective in trending bull markets
#------------------------------------------------------------------------------------------------------------------
class MomentumStrategy(BaseStrategy):
    def __init__(self, tickers=None, lookback_period=50, holding_period=20):
        """
        Initialize the momentum strategy.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols
        lookback_period : int, optional
            Period for momentum calculation
        holding_period : int, optional
            Holding period for positions
        """
        super().__init__(tickers)
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self.position_start_dates = {}  # track when positions were opened
        
    def calculate_momentum(self, data, ticker, lookback_period=None):
        """
        Calculate momentum for a ticker.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
        ticker : str
            Ticker symbol
        lookback_period : int, optional
            Period for momentum calculation
            
        Returns:
        --------
        float
            Momentum score
        """
        if lookback_period is None:
            lookback_period = self.lookback_period
            
        if len(data) < lookback_period:
            return 0
            
        # Get price data
        if ticker in data.columns:
            # Check if the data is a price series or a dataframe
            if isinstance(data[ticker], pd.Series):
                prices = data[ticker]
            elif isinstance(data[ticker], pd.DataFrame) and 'Close' in data[ticker].columns:
                prices = data[ticker]['Close']
            else:
                # Handle the case when data is a Series with mixed data types
                try:
                    prices = pd.to_numeric(data[ticker], errors='coerce')
                    prices = prices.fillna(method='ffill')  # Fill forward any NaNs
                except:
                    return 0  # Return 0 if conversion fails
        else:
            try:
                prices = data[ticker]['Close']
            except:
                return 0  # Return 0 if data not available
            
        # Calculate returns over lookback period
        try:
            start_price = float(prices.iloc[-lookback_period])
            end_price = float(prices.iloc[-1])
            
            if start_price <= 0:
                return 0
                
            # Momentum score
            momentum = (end_price / start_price) - 1
            return momentum
        except (ValueError, TypeError):
            return 0  # Return 0 if conversion fails
        
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
            
        # Check if strategy is suitable for current regime
        if current_regime is not None:
            # Handle the case where current_regime could be a pandas Series or a string
            if isinstance(current_regime, pd.Series):
                # If it's a Series, we need to extract the value
                if len(current_regime) > 0:  # Ensure it's not empty
                    current_regime_value = current_regime.iloc[0]
                else:
                    current_regime_value = None
            else:
                # If it's already a string, use it directly
                current_regime_value = current_regime
                
            # Check if we're in a bearish regime
            if current_regime_value == 'Bearish':
                # Close all positions in bearish regime
                for ticker, quantity in list(self.positions.items()):
                    if quantity > 0:
                        signals.append({
                            'ticker': ticker,
                            'order_type': 'SELL',
                            'quantity': quantity
                        })
                return signals
            
        # Calculate momentum for each ticker
        momentum_scores = {}
        for ticker in self.tickers:
            momentum_scores[ticker] = self.calculate_momentum(data, ticker)
            
        # Sort by momentum
        sorted_tickers = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top momentum stocks to buy
        top_n = min(5, len(sorted_tickers))
        buy_tickers = [ticker for ticker, score in sorted_tickers[:top_n] if score > 0]
        
        # Bottom momentum stocks to sell
        bottom_n = min(5, len(sorted_tickers))
        sell_tickers = [ticker for ticker, score in sorted_tickers[-bottom_n:] if score < 0]
        
        # Close positions for tickers with negative momentum or holding period expired
        current_date_idx = data.index.get_loc(date)
        for ticker, quantity in list(self.positions.items()):
            if ticker in sell_tickers or ticker not in buy_tickers:
                if quantity > 0:
                    signals.append({
                        'ticker': ticker,
                        'order_type': 'SELL',
                        'quantity': quantity
                    })
            elif ticker in self.position_start_dates:
                # Check if holding period expired
                start_date_idx = data.index.get_loc(self.position_start_dates[ticker])
                holding_days = current_date_idx - start_date_idx
                
                if holding_days >= self.holding_period:
                    signals.append({
                        'ticker': ticker,
                        'order_type': 'SELL',
                        'quantity': quantity
                    })
        
        # Open new positions for top momentum tickers
        current_data = data.loc[date]
        available_capital = self.calculate_available_capital(current_data)
        
        for ticker in buy_tickers:
            # Skip if already have a position
            if ticker in self.positions and self.positions[ticker] > 0:
                continue
                
            # Get current price
            if ticker in current_data:
                price = current_data[ticker]
            else:
                price = current_data[ticker]['Close']
                
            # Calculate position size
            quantity = self.calculate_position_size(
                available_capital / len(buy_tickers),
                price,
                risk_pct=0.02,
                stop_loss_pct=0.05
            )
            
            if quantity > 0:
                signals.append({
                    'ticker': ticker,
                    'order_type': 'BUY',
                    'quantity': quantity
                })
                self.position_start_dates[ticker] = date
                
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
            if ticker in current_data:
                price = current_data[ticker]
            else:
                price = current_data[ticker]['Close']
                
            positions_value += price * quantity
            
        # Available capital
        available_capital = total_capital - positions_value
        
        return max(0, available_capital)