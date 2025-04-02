import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .fama_french_strategy import FamaFrenchStrategy


#------------------------------------------------------------------------------------------------------------------
# Regime Adaptive Strategy
# Dynamically switches between strategies based on detected market regime
# Meta-strategy that delegates to specialized sub-strategies
# 
# Strategy logic:
# - Uses momentum strategy in bullish market regimes
# - Uses mean reversion strategy in neutral market regimes
# - Uses defensive factor strategy in bearish market regimes
# - Closes all positions when regime changes
# - Maintains separate tracking for each sub-strategy
#------------------------------------------------------------------------------------------------------------------
class RegimeAdaptiveStrategy(BaseStrategy):
    def __init__(self, tickers=None):
        """
        Initialize the regime adaptive strategy.
        
        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols
        """
        super().__init__(tickers)
        # Initialize sub-strategies
        self.momentum_strategy = MomentumStrategy(tickers)
        self.mean_reversion_strategy = MeanReversionStrategy(tickers)
        self.fama_french_strategy = FamaFrenchStrategy(tickers)
        self.current_strategy = None
        self.current_regime = None
        
    def initialize(self, data):
        """
        Initialize the strategy with market data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data with regime information
        """
        super().initialize(data)
        self.momentum_strategy.initialize(data)
        self.mean_reversion_strategy.initialize(data)
        self.fama_french_strategy.initialize(data)
        
    def update_positions(self, transactions):
        """
        Update positions based on executed transactions.
        
        Parameters:
        -----------
        transactions : list
            List of transaction dictionaries
        """
        super().update_positions(transactions)
        self.momentum_strategy.update_positions(transactions)
        self.mean_reversion_strategy.update_positions(transactions)
        self.fama_french_strategy.update_positions(transactions)
        
    def select_strategy(self, regime):
        """
        Select the appropriate strategy based on market regime.
        
        Parameters:
        -----------
        regime : str or pandas.Series
            Market regime ('Bullish', 'Neutral', 'Bearish')
            
        Returns:
        --------
        object
            Selected strategy object
        """
        # Handle the case where regime could be a pandas Series or a string
        if regime is not None:
            if isinstance(regime, pd.Series):
                # If it's a Series, we need to extract the value
                if len(regime) > 0:  # Ensure it's not empty
                    regime_value = regime.iloc[0]
                else:
                    regime_value = None
            else:
                # If it's already a string, use it directly
                regime_value = regime
                
            if regime_value == 'Bullish':
                return self.momentum_strategy
            elif regime_value == 'Neutral':
                return self.mean_reversion_strategy
            elif regime_value == 'Bearish':
                return self.fama_french_strategy  # more defensive
        
        # Default to mean reversion if regime is unknown or None
        return self.mean_reversion_strategy
        
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
        
        # If no regime information, try to detect it
        if current_regime is None:
            if 'regime' in data.columns:
                current_regime = data['regime'].iloc[-1]
        
        # Process current_regime if it's a Series
        current_regime_value = None
        if current_regime is not None:
            if isinstance(current_regime, pd.Series):
                if len(current_regime) > 0:
                    current_regime_value = current_regime.iloc[0]
            else:
                current_regime_value = current_regime
                
        # Process self.current_regime if it's a Series
        previous_regime_value = None
        if self.current_regime is not None:
            if isinstance(self.current_regime, pd.Series):
                if len(self.current_regime) > 0:
                    previous_regime_value = self.current_regime.iloc[0]
            else:
                previous_regime_value = self.current_regime
        
        # Close all positions if regime has changed
        if (current_regime_value != previous_regime_value) and (previous_regime_value is not None):
            for ticker, quantity in list(self.positions.items()):
                if quantity > 0:
                    signals.append({
                        'ticker': ticker,
                        'order_type': 'SELL',
                        'quantity': quantity
                    })
        
        # Update current regime
        self.current_regime = current_regime
        
        # Select strategy based on regime
        self.current_strategy = self.select_strategy(current_regime)
        
        # Generate signals using the selected strategy
        strategy_signals = self.current_strategy.generate_signals(date, data, **kwargs)
        signals.extend(strategy_signals)
        
        return signals
    
    def get_strategy_allocation(self):
        """
        Get the current strategy allocation based on regime.
        
        Returns:
        --------
        dict
            Allocation weights for each strategy
        """
        allocation = {
            'momentum': 0,
            'mean_reversion': 0,
            'fama_french': 0
        }
        
        # Handle the case where current_regime could be a pandas Series or a string
        if self.current_regime is not None:
            if isinstance(self.current_regime, pd.Series):
                # If it's a Series, we need to extract the value
                if len(self.current_regime) > 0:  # Ensure it's not empty
                    current_regime_value = self.current_regime.iloc[0]
                else:
                    current_regime_value = None
            else:
                # If it's already a string, use it directly
                current_regime_value = self.current_regime
                
            if current_regime_value == 'Bullish':
                allocation['momentum'] = 0.7
                allocation['mean_reversion'] = 0.2
                allocation['fama_french'] = 0.1
            elif current_regime_value == 'Neutral':
                allocation['momentum'] = 0.3
                allocation['mean_reversion'] = 0.5
                allocation['fama_french'] = 0.2
            elif current_regime_value == 'Bearish':
                allocation['momentum'] = 0.1
                allocation['mean_reversion'] = 0.3
                allocation['fama_french'] = 0.6
            else:
                # Default allocation
                allocation['momentum'] = 0.33
                allocation['mean_reversion'] = 0.33
                allocation['fama_french'] = 0.34
        else:
            # Default allocation if no regime information
            allocation['momentum'] = 0.33
            allocation['mean_reversion'] = 0.33
            allocation['fama_french'] = 0.34
        
        return allocation