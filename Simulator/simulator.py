import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


#------------------------------------------------------------------------------------------------------------------
# Market Simulator for Strategy Backtesting
# Handles portfolio allocation, execution, and performance tracking
# Provides realistic simulation of trading strategies with position tracking
# 
# Components:
# - Portfolio management with positions and cash tracking
# - Order execution with volume and capital constraints
# - Performance metrics calculation (returns, drawdowns, Sharpe ratio)
# - Transaction logging and equity curve generation
# - Data fetching and labeling for model training
#------------------------------------------------------------------------------------------------------------------
class Simulator:
    def __init__(self):
        """
        Initialize the simulator.
        """
        self.data = None  # market data
        self.strategy = None  # trading strategy
        self.initial_capital = 100000.0  # starting capital
        self.positions = {}  # current positions
        self.portfolio_value = []  # portfolio value over time
        self.returns = []  # daily returns
        self.transactions = []  # record of all transactions
        self.tickers = []  # list of tickers to track
        self.labeled_data = None  # data with future state labels
        
    def initialize(self, data, strategy, initial_capital=100000.0, tickers=None):
        """
        Initialize the simulator with data and strategy.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Market data with regime information
        strategy : object
            Trading strategy object
        initial_capital : float, optional
            Initial capital for backtesting
        tickers : list, optional
            List of additional tickers to track
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.portfolio_value = []
        self.returns = []
        self.transactions = []
        self.tickers = tickers if tickers is not None else []
        
    def calculate_portfolio_value(self, date):
        """
        Calculate portfolio value at given date.
        
        Parameters:
        -----------
        date : datetime
            Date to calculate value for
            
        Returns:
        --------
        float
            Portfolio value
        """
        # Get prices at current date
        current_data = self.data.loc[date]
        
        # Calculate positions value
        positions_value = 0
        for ticker, quantity in self.positions.items():
            if ticker in current_data:
                price = current_data[ticker] if isinstance(current_data[ticker], (int, float)) else current_data[ticker]['Close']
                positions_value += price * quantity
            
        # Total portfolio value
        total_value = self.cash + positions_value
        return total_value
        
    def execute_order(self, date, ticker, order_type, quantity, price):
        """
        Execute a trading order.
        
        Parameters:
        -----------
        date : datetime
            Order date
        ticker : str
            Ticker symbol
        order_type : str
            'BUY' or 'SELL'
        quantity : int
            Number of shares
        price : float
            Execution price
            
        Returns:
        --------
        bool
            True if order executed successfully, False otherwise
        """
        order_value = price * quantity
        
        if order_type == 'BUY':
            # Check if enough cash
            if order_value > self.cash:
                # Adjust quantity based on available cash
                quantity = int(self.cash / price)
                if quantity <= 0:
                    return False
                order_value = price * quantity
                
            # Update cash and positions
            self.cash -= order_value
            if ticker in self.positions:
                self.positions[ticker] += quantity
            else:
                self.positions[ticker] = quantity
                
        elif order_type == 'SELL':
            # Check if position exists
            if ticker not in self.positions or self.positions[ticker] < quantity:
                if ticker in self.positions:
                    # Sell what we have
                    quantity = self.positions[ticker]
                    order_value = price * quantity
                else:
                    return False
                    
            # Update cash and positions
            self.cash += order_value
            self.positions[ticker] -= quantity
            if self.positions[ticker] == 0:
                del self.positions[ticker]
                
        # Record transaction
        self.transactions.append({
            'date': date,
            'ticker': ticker,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'value': order_value
        })
        
        return True
        
    def run(self):
        """
        Run the backtest simulation.
        
        Returns:
        --------
        dict
            Backtest results
        """
        if self.data is None or self.strategy is None:
            print("Simulator not initialized. Call initialize() first.")
            return None
            
        # Initialize strategy
        self.strategy.initialize(self.data)
        
        # Initialize results
        portfolio_values = []
        returns = []
        equity_curve = []
        current_value = self.initial_capital
        
        # Run simulation day by day
        for date, data_row in self.data.iterrows():
            # Generate signals
            signals = self.strategy.generate_signals(date, self.data.loc[:date], 
                                                    current_regime=data_row.get('regime'))
            
            # Execute signals
            for signal in signals:
                ticker = signal['ticker']
                order_type = signal['order_type']
                quantity = signal['quantity']
                price = data_row[ticker] if isinstance(data_row[ticker], (int, float)) else data_row[ticker]['Close']
                
                self.execute_order(date, ticker, order_type, quantity, price)
            
            # Calculate portfolio value
            current_value = self.calculate_portfolio_value(date)
            portfolio_values.append(current_value)
            
            # Calculate return
            if len(portfolio_values) > 1:
                daily_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
            else:
                daily_return = 0
                
            returns.append(daily_return)
            
        # Calculate equity curve
        equity_curve = pd.Series(portfolio_values, index=self.data.index)
        returns_series = pd.Series(returns, index=self.data.index)
        
        # Calculate performance metrics
        total_return = (portfolio_values[-1] / self.initial_capital) - 1
        annual_return = ((1 + total_return) ** (252 / len(returns))) - 1 if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        win_rate = self.calculate_win_rate(returns_series)
        
        # Store results
        results = {
            'equity_curve': equity_curve,
            'returns': returns_series,
            'positions': self.positions,
            'transactions': pd.DataFrame(self.transactions),
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        
        return results
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown.
        
        Parameters:
        -----------
        equity_curve : pandas.Series
            Portfolio value over time
            
        Returns:
        --------
        float
            Maximum drawdown as a percentage
        """
        # Calculate drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        
        # Get maximum drawdown
        max_drawdown = drawdown.min()
        
        return max_drawdown
    
    def calculate_win_rate(self, returns):
        """
        Calculate win rate.
        
        Parameters:
        -----------
        returns : pandas.Series
            Daily returns
            
        Returns:
        --------
        float
            Win rate as a percentage
        """
        if len(returns) == 0:
            return 0
            
        wins = (returns > 0).sum()
        win_rate = wins / len(returns)
        
        return win_rate
        
    def fetch_ticker_data(self, ticker, start_date, end_date=None, include_future_days=0):
        """
        Fetch historical data for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to fetch
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format. Defaults to current date.
        include_future_days : int, optional
            Number of future days to include for labeling. Defaults to 0.
            
        Returns:
        --------
        pandas.DataFrame
            Historical price data for the ticker
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # If we need future data for labeling, extend the end date
        if include_future_days > 0:
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                extended_end_date = (end_date_obj + timedelta(days=include_future_days + 5)).strftime('%Y-%m-%d')
                # Adding 5 extra days to account for weekends and holidays
                ticker_data = yf.download(ticker, start=start_date, end=extended_end_date)
            except Exception as e:
                print(f"Error fetching extended data for {ticker}: {e}")
                ticker_data = yf.download(ticker, start=start_date, end=end_date)
        else:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)
            
        return ticker_data
    
    def label_data_with_future_states(self, data, future_window=5, threshold=0.02):
        """
        Label data with future market states for training models.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Price data to label
        future_window : int, optional
            Number of days to look ahead for labeling. Defaults to 5.
        threshold : float, optional
            Threshold for determining good/bad performance. Defaults to 0.02 (2%).
            
        Returns:
        --------
        pandas.DataFrame
            Data with added future state labels
        """
        if data is None or len(data) == 0:
            print("No data available for labeling.")
            return None
            
        labeled_data = data.copy()
        
        # Calculate future returns
        labeled_data['future_return'] = labeled_data['Close'].pct_change(future_window).shift(-future_window)
        
        # Label based on future performance on -1 to 1 scale
        labeled_data['future_state'] = 'neutral'
        labeled_data.loc[labeled_data['future_return'] > threshold, 'future_state'] = 'good'
        labeled_data.loc[labeled_data['future_return'] < -threshold, 'future_state'] = 'bad'
        
        # Create numeric state for modeling on -1 to 1 scale
        state_map = {'bad': -1, 'neutral': 0, 'good': 1}
        labeled_data['future_state_num'] = labeled_data['future_state'].map(state_map)
        
        self.labeled_data = labeled_data
        return labeled_data
    
    def get_training_data(self, ticker, start_date, end_date=None, future_window=5, threshold=0.02):
        """
        Get labeled data for a specific ticker for model training.
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to fetch
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format. Defaults to current date.
        future_window : int, optional
            Number of days to look ahead for labeling. Defaults to 5.
        threshold : float, optional
            Threshold for determining good/bad performance. Defaults to 0.02 (2%).
            
        Returns:
        --------
        pandas.DataFrame
            Labeled data ready for model training
        """
        # Fetch data including future days
        ticker_data = self.fetch_ticker_data(ticker, start_date, end_date, include_future_days=future_window)
        
        # Add technical indicators for features
        ticker_data['returns'] = ticker_data['Close'].pct_change()
        ticker_data['volatility'] = ticker_data['returns'].rolling(window=20).std()
        ticker_data['sma_50'] = ticker_data['Close'].rolling(window=50).mean()
        ticker_data['sma_200'] = ticker_data['Close'].rolling(window=200).mean()
        ticker_data['rsi'] = self._calculate_rsi(ticker_data['Close'])
        
        # Create normalized price features (on -1 to 1 scale)
        ticker_data['price_sma50_ratio'] = (ticker_data['Close'] / ticker_data['sma_50'] - 1) * 5  # Scale to approx -1 to 1
        ticker_data['sma50_sma200_ratio'] = (ticker_data['sma_50'] / ticker_data['sma_200'] - 1) * 5  # Scale to approx -1 to 1
        
        # Normalize RSI to -1 to 1 scale (from 0-100 scale)
        ticker_data['rsi_norm'] = (ticker_data['rsi'] - 50) / 50  # Now -1 to 1
        
        # Scale volatility to approximate -1 to 1 range
        mean_vol = ticker_data['volatility'].mean()
        ticker_data['volatility_norm'] = ticker_data['volatility'] / (2 * mean_vol) - 0.5
        ticker_data['volatility_norm'] = ticker_data['volatility_norm'].clip(-1, 1)  # Ensure within range
        
        # Label with future states
        labeled_data = self.label_data_with_future_states(ticker_data, future_window, threshold)
        
        # Remove rows with NaN values
        labeled_data = labeled_data.dropna()
        
        return labeled_data
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculate the RSI technical indicator.
        
        Parameters:
        -----------
        prices : pandas.Series
            Price series
        window : int, optional
            RSI window. Defaults to 14.
            
        Returns:
        --------
        pandas.Series
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def generate_training_samples(self, market_ticker='SPY', additional_tickers=None, 
                                  start_date='2018-01-01', end_date=None, 
                                  future_window=5, threshold=0.02, n_samples=100):
        """
        Generate random labeled samples for model training.
        
        Parameters:
        -----------
        market_ticker : str, optional
            Market benchmark ticker. Defaults to 'SPY'.
        additional_tickers : list, optional
            List of additional tickers to include in the dataset.
        start_date : str, optional
            Start date in YYYY-MM-DD format. Defaults to '2018-01-01'.
        end_date : str, optional
            End date in YYYY-MM-DD format. Defaults to current date.
        future_window : int, optional
            Number of days to look ahead for labeling. Defaults to 5.
        threshold : float, optional
            Threshold for determining good/bad performance. Defaults to 0.02 (2%).
        n_samples : int, optional
            Number of random samples to generate. Defaults to 100.
            
        Returns:
        --------
        dict
            Dictionary with training data for all tickers and combined features
        """
        # Get market data with regime labels
        market_data = self.get_training_data(market_ticker, start_date, end_date, 
                                           future_window, threshold)
        
        if market_data is None or len(market_data) == 0:
            print("Failed to get market data for training.")
            return None
            
        # Initialize dictionary to store all ticker data
        training_data = {
            'market': market_data,
            'tickers': {}
        }
        
        # Process additional tickers if provided
        if additional_tickers:
            for ticker in additional_tickers:
                try:
                    ticker_data = self.get_training_data(ticker, start_date, end_date, 
                                                      future_window, threshold)
                    if ticker_data is not None and len(ticker_data) > 0:
                        # Align ticker data with market data to ensure consistent dates
                        common_dates = ticker_data.index.intersection(market_data.index)
                        if len(common_dates) > 0:
                            aligned_ticker_data = ticker_data.loc[common_dates]
                            training_data['tickers'][ticker] = aligned_ticker_data
                            print(f"Processed {ticker}: {len(aligned_ticker_data)} aligned trading days")
                        else:
                            print(f"Warning: No common trading days between {market_ticker} and {ticker}")
                except Exception as e:
                    print(f"Error processing ticker {ticker}: {e}")
        
        # Generate random samples from the aligned data
        if n_samples > 0 and len(market_data) > n_samples:
            # Only sample from dates that have data for all tickers
            valid_dates = set(market_data.index)
            for ticker_data in training_data['tickers'].values():
                valid_dates = valid_dates.intersection(ticker_data.index)
            
            valid_dates = sorted(list(valid_dates))
            print(f"Found {len(valid_dates)} trading days with data for all tickers")
            
            if len(valid_dates) > 0:
                # Get valid indices for sampling
                if len(valid_dates) < n_samples:
                    print(f"Warning: Only {len(valid_dates)} valid samples available (less than requested {n_samples})")
                    n_samples = len(valid_dates)
                
                # Get random indices for sampling
                sample_indices = np.random.choice(valid_dates, size=n_samples, replace=False)
                
                # Create sample dataset with only valid dates
                samples = market_data.loc[sample_indices].copy()
                
                # Add samples to training data
                training_data['samples'] = samples
            
            # Create feature matrix for machine learning (all on -1 to 1 scale)
            features = []
            for idx, row in samples.iterrows():
                feature_dict = {
                    'date': idx,
                    'market_regime': row.get('regime', 'unknown'),
                    'market_regime_score': row.get('regime_score', 0),  # -1 to 1 scale
                    'market_returns': row['returns'].clip(-1, 1),  # Clip extreme values
                    'market_volatility': row['volatility_norm'],  # Already normalized
                    'market_rsi': row['rsi_norm'],  # Already normalized
                    'market_price_trend': row['price_sma50_ratio'].clip(-1, 1),  # Price relative to 50-day SMA
                    'market_trend_strength': row['sma50_sma200_ratio'].clip(-1, 1),  # Trend strength indicator
                    'future_state': row['future_state'],
                    'future_state_num': row['future_state_num'],  # Already -1 to 1
                    'future_return': row['future_return']
                }
                
                # Add features from additional tickers if available (all normalized to -1 to 1)
                for ticker, ticker_data in training_data['tickers'].items():
                    if idx in ticker_data.index:
                        ticker_row = ticker_data.loc[idx]
                        feature_dict[f'{ticker}_returns'] = ticker_row['returns'].clip(-1, 1)
                        feature_dict[f'{ticker}_volatility'] = ticker_row['volatility_norm']
                        feature_dict[f'{ticker}_rsi'] = ticker_row['rsi_norm']
                        feature_dict[f'{ticker}_price_trend'] = ticker_row['price_sma50_ratio'].clip(-1, 1)
                        feature_dict[f'{ticker}_trend_strength'] = ticker_row['sma50_sma200_ratio'].clip(-1, 1)
                        feature_dict[f'{ticker}_future_return'] = ticker_row['future_return']
                
                features.append(feature_dict)
            
            # Convert features to DataFrame
            feature_df = pd.DataFrame(features)
            training_data['feature_matrix'] = feature_df
        
        return training_data
        
    def get_current_market_features(self, market_ticker='SPY', additional_tickers=None, lookback_days=252):
        """
        Get current market features for use with trained models.
        This is designed for production use to get the latest market state.
        
        Parameters:
        -----------
        market_ticker : str, optional
            Market benchmark ticker. Defaults to 'SPY'.
        additional_tickers : list, optional
            List of additional tickers to include. Defaults to None.
        lookback_days : int, optional
            Number of days of historical data to fetch for calculating features.
        
        Returns:
        --------
        dict
            Dictionary of current market features normalized to -1 to 1 scale
        """
        try:
            # Calculate dates
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Fetch market data
            market_data = self.fetch_ticker_data(market_ticker, start_date, end_date)
            
            if market_data is None or len(market_data) == 0:
                print(f"Failed to get market data for {market_ticker}")
                return None
            
            # Calculate standard features
            market_data['returns'] = market_data['Close'].pct_change()
            market_data['volatility'] = market_data['returns'].rolling(window=20).std()
            market_data['sma_50'] = market_data['Close'].rolling(window=50).mean()
            market_data['sma_200'] = market_data['Close'].rolling(window=200).mean()
            market_data['rsi'] = self._calculate_rsi(market_data['Close'])
            
            # Normalize features to -1 to 1 scale
            market_data['returns_norm'] = market_data['returns'].clip(-1, 1)
            market_data['price_sma50_ratio'] = (market_data['Close'] / market_data['sma_50'] - 1) * 5
            market_data['price_sma50_ratio'] = market_data['price_sma50_ratio'].clip(-1, 1)
            market_data['sma50_sma200_ratio'] = (market_data['sma_50'] / market_data['sma_200'] - 1) * 5
            market_data['sma50_sma200_ratio'] = market_data['sma50_sma200_ratio'].clip(-1, 1)
            market_data['rsi_norm'] = (market_data['rsi'] - 50) / 50
            
            # Scale volatility
            mean_vol = market_data['volatility'].mean()
            market_data['volatility_norm'] = market_data['volatility'] / (2 * mean_vol) - 0.5
            market_data['volatility_norm'] = market_data['volatility_norm'].clip(-1, 1)
            
            # Get the latest row (current market state)
            latest_data = market_data.iloc[-1].copy()
            
            # Create feature dictionary
            current_features = {
                'date': latest_data.name,  # Index is the date
                'market_returns': latest_data['returns_norm'],
                'market_volatility': latest_data['volatility_norm'],
                'market_rsi': latest_data['rsi_norm'],
                'market_price_trend': latest_data['price_sma50_ratio'],
                'market_trend_strength': latest_data['sma50_sma200_ratio']
            }
            
            # Add additional tickers if provided
            if additional_tickers:
                for ticker in additional_tickers:
                    try:
                        # Get ticker data
                        ticker_data = self.fetch_ticker_data(ticker, start_date, end_date)
                        
                        if ticker_data is not None and len(ticker_data) > 0:
                            # Calculate and normalize features
                            ticker_data['returns'] = ticker_data['Close'].pct_change()
                            ticker_data['volatility'] = ticker_data['returns'].rolling(window=20).std()
                            ticker_data['sma_50'] = ticker_data['Close'].rolling(window=50).mean()
                            ticker_data['sma_200'] = ticker_data['Close'].rolling(window=200).mean()
                            ticker_data['rsi'] = self._calculate_rsi(ticker_data['Close'])
                            
                            ticker_data['returns_norm'] = ticker_data['returns'].clip(-1, 1)
                            ticker_data['price_sma50_ratio'] = (ticker_data['Close'] / ticker_data['sma_50'] - 1) * 5
                            ticker_data['price_sma50_ratio'] = ticker_data['price_sma50_ratio'].clip(-1, 1)
                            ticker_data['sma50_sma200_ratio'] = (ticker_data['sma_50'] / ticker_data['sma_200'] - 1) * 5
                            ticker_data['sma50_sma200_ratio'] = ticker_data['sma50_sma200_ratio'].clip(-1, 1)
                            ticker_data['rsi_norm'] = (ticker_data['rsi'] - 50) / 50
                            
                            mean_vol = ticker_data['volatility'].mean()
                            ticker_data['volatility_norm'] = ticker_data['volatility'] / (2 * mean_vol) - 0.5
                            ticker_data['volatility_norm'] = ticker_data['volatility_norm'].clip(-1, 1)
                            
                            # Get latest data
                            latest_ticker = ticker_data.iloc[-1]
                            
                            # Add to features
                            current_features[f'{ticker}_returns'] = latest_ticker['returns_norm']
                            current_features[f'{ticker}_volatility'] = latest_ticker['volatility_norm']
                            current_features[f'{ticker}_rsi'] = latest_ticker['rsi_norm']
                            current_features[f'{ticker}_price_trend'] = latest_ticker['price_sma50_ratio']
                            current_features[f'{ticker}_trend_strength'] = latest_ticker['sma50_sma200_ratio']
                    except Exception as e:
                        print(f"Error processing ticker {ticker}: {e}")
                        
            return current_features
            
        except Exception as e:
            print(f"Error getting current market features: {e}")
            return None