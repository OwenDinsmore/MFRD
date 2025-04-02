import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


#------------------------------------------------------------------------------------------------------------------
# Mean Reversion Strategy
# Buys oversold stocks and sells overbought stocks
# Performs well in neutral/sideways markets
# 
# Strategy logic:
# - Buy assets with RSI below oversold threshold (typically 30)
# - Sell assets with RSI above overbought threshold (typically 70)
# - Use shorter holding periods than momentum strategies
# - Adjust position sizing based on market regime
# - Particularly effective in range-bound/sideways markets
#------------------------------------------------------------------------------------------------------------------
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, tickers=None, rsi_period=14, overbought=70, oversold=30, holding_period=5):
        """
        Initialize the mean reversion strategy.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols
        rsi_period : int, optional
            Period for RSI calculation
        overbought : int, optional
            RSI threshold for overbought condition
        oversold : int, optional
            RSI threshold for oversold condition
        holding_period : int, optional
            Holding period for positions
        """
        super().__init__(tickers)
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.holding_period = holding_period
        self.position_start_dates = {}  # track when positions were opened
        
    def calculate_rsi(self, data, ticker, period=None):
        """
        Calculate RSI for a ticker.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data
        ticker : str
            Ticker symbol
        period : int, optional
            Period for RSI calculation
            
        Returns:
        --------
        float
            RSI value
        """
        if period is None:
            period = self.rsi_period
            
        if len(data) <= period:
            return 50  # neutral
            
        try:
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
                        return 50  # Return neutral RSI if conversion fails
            else:
                try:
                    prices = data[ticker]['Close']
                except:
                    return 50  # Return neutral RSI if data not available
                
            # Convert prices to numeric if needed
            prices = pd.to_numeric(prices, errors='coerce')
                
            # Calculate returns
            delta = prices.diff()
            
            # Calculate up and down moves
            up_moves = delta.copy()
            up_moves[up_moves < 0] = 0
            
            down_moves = abs(delta.copy())
            down_moves[down_moves <= 0] = 0
            
            # Calculate average up and down moves
            avg_up = up_moves.rolling(window=period).mean()
            avg_down = down_moves.rolling(window=period).mean()
            
            # Calculate relative strength and RSI
            rs = avg_up / avg_down
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
        except Exception as e:
            # Return neutral RSI value in case of any error
            return 50
        
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
        if len(data) <= self.rsi_period:
            return signals
            
        # Check if strategy is suitable for current regime
        # Mean reversion works better in neutral markets
        # and worse in strong trending markets
        
        # Handle the case where current_regime could be a pandas Series or a string
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
                
            if current_regime_value == 'Bullish':
                # Reduce position size in bullish regime - momentum works better
                risk_modifier = 0.5
            elif current_regime_value == 'Bearish':
                # More cautious in bearish regimes
                risk_modifier = 0.3
            else:
                # Normal risk in neutral regime
                risk_modifier = 1.0
        else:
            # Default to normal risk if no regime information
            risk_modifier = 1.0
            
        # Calculate RSI for each ticker
        rsi_values = {}
        for ticker in self.tickers:
            rsi_values[ticker] = self.calculate_rsi(data, ticker)
            
        # Find oversold and overbought tickers
        oversold_tickers = [ticker for ticker, rsi in rsi_values.items() if rsi <= self.oversold]
        overbought_tickers = [ticker for ticker, rsi in rsi_values.items() if rsi >= self.overbought]
        
        # Close positions for overbought tickers (if we're long) or holding period expired
        current_date_idx = data.index.get_loc(date)
        for ticker, quantity in list(self.positions.items()):
            if ticker in overbought_tickers and quantity > 0:
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
        
        # Open new positions for oversold tickers
        current_data = data.loc[date]
        available_capital = self.calculate_available_capital(current_data)
        
        for ticker in oversold_tickers:
            # Skip if already have a position
            if ticker in self.positions and self.positions[ticker] > 0:
                continue
                
            # Get current price
            if ticker in current_data:
                price = current_data[ticker]
            else:
                price = current_data[ticker]['Close']
                
            # Calculate position size with risk modifier
            quantity = self.calculate_position_size(
                available_capital / len(oversold_tickers) * risk_modifier,
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