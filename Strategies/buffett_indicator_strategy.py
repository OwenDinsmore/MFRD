import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .base_strategy import BaseStrategy

#------------------------------------------------------------------------------------------------------------------
# Buffett Indicator Strategy
# This strategy uses the Buffett Indicator (total market value / GDP) to gauge market valuation.
# It calculates the daily ratio, then compares it to a historical trend (mean and standard deviation)
# computed over a specified lookback period. If the current ratio is significantly above the trend,
# it suggests that the market is overvalued, triggering a SELL signal for the tracked market ticker.
# Conversely, if the ratio is significantly below the trend, it suggests that the market is undervalued,
# triggering a BUY signal. In between the thresholds, no signal is generated.
#
# Assumptions:
# - The input data is a pandas.DataFrame containing daily data with at least two columns:
#   "total_market_value" and "GDP". These values should be such that the ratio is computed as:
#   ratio = (total_market_value / GDP) * 100.
# - The strategy is intended to trade a single market ticker (default: "SPY") representing the overall market.
#------------------------------------------------------------------------------------------------------------------
class BuffettIndicatorStrategy(BaseStrategy):
    def __init__(self, tickers=None, lookback_period=500, overvalued_threshold=2.0, undervalued_threshold=-2.0):
        """
        Initialize the Buffett Indicator strategy.

        Parameters:
        -----------
        tickers : list, optional
            List of ticker symbols to trade. If not provided, defaults to ["SPY"].
        lookback_period : int, optional
            Number of days to use for calculating the historical trend.
        overvalued_threshold : float, optional
            Z-score above which the market is considered strongly overvalued.
        undervalued_threshold : float, optional
            Z-score below which the market is considered strongly undervalued.
        """
        # If no ticker is provided, default to a market index, e.g. SPY.
        if tickers is None or not tickers:
            tickers = ["SPY"]
        super().__init__(tickers)
        self.lookback_period = lookback_period
        self.overvalued_threshold = overvalued_threshold
        self.undervalued_threshold = undervalued_threshold

    def calculate_indicator(self, data):
        """
        Calculate the Buffett Indicator as the ratio of total_market_value to GDP (in percent).

        Parameters:
        -----------
        data : pandas.DataFrame
            Market data containing "total_market_value" and "GDP" columns.

        Returns:
        --------
        float
            Buffett Indicator for the provided data row.
        """
        try:
            total_market_value = data["total_market_value"]
            gdp = data["GDP"]
            # Compute the ratio and convert it into percentage
            ratio = (total_market_value / gdp) * 100
            return ratio
        except Exception as e:
            # In case of error, return a neutral value (e.g. 0)
            return 0

    def calculate_available_capital(self, current_data, total_capital=100000.0):
        """
        Calculate available capital based on current positions.

        Parameters:
        -----------
        current_data : pandas.Series
            Current market data.
        total_capital : float, optional
            Total capital available.

        Returns:
        --------
        float
            Available capital.
        """
        positions_value = 0
        for ticker, quantity in self.positions.items():
            try:
                if ticker in current_data:
                    price = current_data[ticker]
                else:
                    price = current_data[ticker]['Close']
                positions_value += price * quantity
            except Exception:
                continue
        available_capital = total_capital - positions_value
        return max(0, available_capital)

    def generate_signals(self, date, data, **kwargs):
        """
        Generate daily trading signals based on the Buffett Indicator.

        Parameters:
        -----------
        date : datetime
            Current date.
        data : pandas.DataFrame
            Market data with daily records containing "total_market_value" and "GDP" columns.
        kwargs : dict
            Additional parameters (if any).

        Returns:
        --------
        list
            List of signal dictionaries.
        """
        signals = []
        
        # Ensure we have enough historical data for trend calculation
        if len(data) < self.lookback_period:
            return signals

        # Get current day's data row
        try:
            current_data = data.loc[date]
        except Exception:
            return signals

        # Calculate today's Buffett Indicator
        current_ratio = self.calculate_indicator(current_data)

        # Calculate historical ratios over the lookback period
        historical_data = data.iloc[-self.lookback_period:]
        historical_ratios = historical_data.apply(self.calculate_indicator, axis=1)
        
        # Compute historical mean and standard deviation
        mean_ratio = historical_ratios.mean()
        std_ratio = historical_ratios.std()

        # Avoid division by zero
        if std_ratio == 0:
            z_score = 0
        else:
            z_score = (current_ratio - mean_ratio) / std_ratio

        # For simplicity, we trade a single market ticker (self.tickers[0])
        ticker = self.tickers[0]

        # Get current price for the ticker, similar logic as in other strategies
        try:
            if ticker in current_data:
                price = current_data[ticker]
                # In case the price is a pandas.Series, try to convert to numeric
                if isinstance(price, pd.Series):
                    price = pd.to_numeric(price, errors='coerce')
                    price = float(price) if not pd.isna(price) else None
            else:
                price = float(current_data[ticker]['Close'])
        except Exception:
            price = None

        # If price is invalid, return no signals
        if price is None or price <= 0:
            return signals

        # Decision logic:
        # 1. If the indicator is strongly overvalued (z_score >= overvalued_threshold) and a long position exists,
        #    then exit the position.
        # 2. If the indicator is strongly undervalued (z_score <= undervalued_threshold) and no position exists,
        #    then enter a long position.
        if z_score >= self.overvalued_threshold:
            # Market is strongly overvalued: if we hold the ticker, generate a SELL signal
            if ticker in self.positions and self.positions[ticker] > 0:
                signals.append({
                    'ticker': ticker,
                    'order_type': 'SELL',
                    'quantity': self.positions[ticker]
                })
        elif z_score <= self.undervalued_threshold:
            # Market is strongly undervalued: if we do not hold the ticker, generate a BUY signal
            if ticker not in self.positions or self.positions[ticker] == 0:
                available_capital = self.calculate_available_capital(current_data)
                quantity = self.calculate_position_size(
                    available_capital, price,
                    risk_pct=0.02, stop_loss_pct=0.05
                )
                if quantity > 0:
                    signals.append({
                        'ticker': ticker,
                        'order_type': 'BUY',
                        'quantity': quantity
                    })

        # Optionally, add logging of the calculated z_score for debugging purposes
        # e.g., print(f"{date}: Buffett Ratio = {current_ratio:.2f}, Mean = {mean_ratio:.2f}, std = {std_ratio:.2f}, z_score = {z_score:.2f}")
        
        return signals