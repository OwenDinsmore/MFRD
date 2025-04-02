import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


#------------------------------------------------------------------------------------------------------------------
# Base Strategy Class
# Abstract base class for all trading strategies
# Provides common functionality and interface for strategy implementations
# 
# Core functions:
# - Strategy initialization with market data
# - Signal generation based on market conditions
# - Position management and tracking
# - Risk-based position sizing
#------------------------------------------------------------------------------------------------------------------
class BaseStrategy(ABC):
    def __init__(self, tickers=None):
        """
        Initialize the strategy.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols
        """
        self.tickers = tickers if tickers is not None else []
        self.data = None
        self.positions = {}  # current positions
        
    def initialize(self, data):
        """
        Initialize the strategy with market data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data with regime information
        """
        self.data = data
        
        # If tickers not specified, extract from data columns
        if not self.tickers and isinstance(data, pd.DataFrame):
            # Look for ticker columns in data
            for col in data.columns:
                if col not in ['returns', 'volatility', 'sma_50', 'sma_200', 'rsi', 'regime', 'regime_prob']:
                    self.tickers.append(col)
    
    @abstractmethod
    def generate_signals(self, date, data, **kwargs):
        """
        Generate trading signals for given date.
        
        Parameters:
        -----------
        date : datetime
            Current date
        data : pandas.DataFrame
            Market data up to current date
        kwargs : dict
            Additional parameters
            
        Returns:
        --------
        list
            List of signal dictionaries
        """
        pass
    
    def update_positions(self, transactions):
        """
        Update positions based on executed transactions.
        
        Parameters:
        -----------
        transactions : list
            List of transaction dictionaries
        """
        for transaction in transactions:
            ticker = transaction['ticker']
            order_type = transaction['order_type']
            quantity = transaction['quantity']
            
            if order_type == 'BUY':
                if ticker in self.positions:
                    self.positions[ticker] += quantity
                else:
                    self.positions[ticker] = quantity
            elif order_type == 'SELL':
                if ticker in self.positions:
                    self.positions[ticker] -= quantity
                    if self.positions[ticker] <= 0:
                        del self.positions[ticker]
    
    def calculate_position_size(self, capital, price, risk_pct=0.02, stop_loss_pct=0.05):
        """
        Calculate position size based on risk.
        
        Parameters:
        -----------
        capital : float
            Available capital
        price : float
            Current price
        risk_pct : float, optional
            Risk percentage of capital (0.02 = 2%)
        stop_loss_pct : float, optional
            Stop loss percentage (0.05 = 5%)
            
        Returns:
        --------
        int
            Number of shares to buy/sell
        """
        risk_amount = capital * risk_pct
        dollar_risk_per_share = price * stop_loss_pct
        
        if dollar_risk_per_share <= 0:
            return 0
            
        shares = int(risk_amount / dollar_risk_per_share)
        
        # Ensure shares doesn't exceed capital
        max_shares = int(capital / price)
        shares = min(shares, max_shares)
        
        return shares