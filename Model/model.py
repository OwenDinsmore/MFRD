import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HMM.hmm import MarketRegimeHMM
from Strategies.momentum_strategy import MomentumStrategy
from Strategies.mean_reversion_strategy import MeanReversionStrategy
from Strategies.fama_french_strategy import FamaFrenchStrategy
from Strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from Strategies.news_llm_strategy import LLMNewsStrategy
from Simulator.simulator import Simulator


#------------------------------------------------------------------------------------------------------------------
# Core Model for Multi-Factor Regime Detection
# Integrates market regime detection with trading strategies and backtesting functionality
# Handles data loading, preprocessing, regime detection, and strategy evaluation
# 
# Data flow:
# Raw price data -> Preprocessing -> Regime Detection -> Strategy Application -> Performance Analysis
#------------------------------------------------------------------------------------------------------------------
class Model:
    def __init__(self, data=None):
        """
        Initialize the model with optional data.
        
        Parameters:
        -----------
        data : pandas.DataFrame, optional
            Historical price data with OHLCV columns
        """
        self.data = data
        self.regime_model = MarketRegimeHMM()
        self.strategies = {}
        self.simulator = Simulator()
        self.results = {}
        self.current_regime = None
        
    def load_data(self, ticker, start_date, end_date=None):
        """
        Load market data for the specified ticker and date range.
        
        Parameters:
        -----------
        ticker : str
            Stock symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format. Defaults to current date.
        """
        try:
            import yfinance as yf
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
                
            self.data = yf.download(ticker, start=start_date, end=end_date)
            print(f"Loaded data for {ticker} from {start_date} to {end_date}")
            return self.data
        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
            
    def preprocess_data(self):
        """
        Preprocess the data for modeling.
        
        Returns:
        --------
        pandas.DataFrame
            Processed data
        """
        if self.data is None:
            print("No data available. Load data first.")
            return None
            
        processed_data = self.data.copy()  # create copy to avoid modifying original
        
        processed_data['returns'] = processed_data['Close'].pct_change()  # calculate daily returns
        
        processed_data['volatility'] = processed_data['returns'].rolling(window=20).std()  # 20-day volatility
        
        processed_data['sma_50'] = processed_data['Close'].rolling(window=50).mean()  # 50-day moving average
        processed_data['sma_200'] = processed_data['Close'].rolling(window=200).mean()  # 200-day moving average
        
        # calculate RSI (relative strength index)
        delta = processed_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        processed_data['rsi'] = 100 - (100 / (1 + rs))
        
        processed_data = processed_data.dropna()  # remove rows with NaN values
        
        return processed_data
    
    def detect_regimes(self):
        """
        Detect market regimes in the data.
        
        Returns:
        --------
        pandas.DataFrame
            Data with regime labels
        """
        if self.data is None:
            print("No data available. Load data first.")
            return None
            
        processed_data = self.preprocess_data()
        if processed_data is None or len(processed_data) == 0:
            return None
            
        self.regime_model.fit(processed_data)  # fit model to detect regimes
        
        states = self.regime_model.states  # get detected regime states
        
        regime_data = processed_data.copy()  # create data with regime labels
        regime_data['regime'] = [self.regime_model.state_labels[s] for s in states]
        regime_data['regime_prob'] = [self.regime_model.state_probabilities[i][state] 
                                     for i, state in enumerate(states)]
        
        # Add regime scores on -1 to 1 scale
        regime_data['regime_score'] = [self.regime_model.get_regime_score(self.regime_model.state_probabilities[i]) 
                                      for i in range(len(states))]
        
        self.current_regime = self.regime_model.get_current_regime()  # get current regime info
        
        return regime_data
    
    def register_strategy(self, name, strategy_class, **params):
        """
        Register a trading strategy.
        
        Parameters:
        -----------
        name : str
            Name for the strategy
        strategy_class : class
            Strategy class to use
        params : dict
            Parameters for the strategy
        """
        self.strategies[name] = strategy_class(**params)
        
    def backtest_all_strategies(self, initial_capital=100000.0):
        """
        Backtest all registered strategies.
        
        Parameters:
        -----------
        initial_capital : float
            Initial capital for backtesting
            
        Returns:
        --------
        dict
            Results for all strategies
        """
        if not self.strategies:
            print("No strategies registered. Register strategies first.")
            return {}
            
        regime_data = self.detect_regimes()
        if regime_data is None:
            return {}
            
        results = {}
        for name, strategy in self.strategies.items():
            print(f"Backtesting {name}...")
            self.simulator.initialize(regime_data, strategy, initial_capital)
            strategy_results = self.simulator.run()
            results[name] = strategy_results
            
        self.results = results
        return results
    
    def plot_strategy_performance(self, benchmark=None):
        """
        Plot performance of all backtested strategies.
        
        Parameters:
        -----------
        benchmark : str, optional
            Ticker symbol for benchmark
            
        Returns:
        --------
        matplotlib.figure.Figure
            Performance comparison figure
        """
        if not self.results:
            print("No backtest results available. Run backtest first.")
            return None
            
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for name, result in self.results.items():  # plot strategy equity curves
            ax.plot(result['equity_curve'], label=name)
            
        if benchmark and self.data is not None:  # plot benchmark for comparison
            benchmark_returns = self.data['Close'].pct_change().dropna()
            benchmark_equity = (1 + benchmark_returns).cumprod() * 100000
            ax.plot(benchmark_equity, label=f"{benchmark} Buy & Hold", linestyle='--')
            
        regime_data = self.detect_regimes()  # highlight regime periods
        if regime_data is not None:
            regime_changes = regime_data['regime'].ne(regime_data['regime'].shift()).cumsum()
            for regime, group in regime_data.groupby(regime_changes):
                color = 'green' if group['regime'].iloc[0] == 'Bullish' else \
                        'red' if group['regime'].iloc[0] == 'Bearish' else 'yellow'
                ax.axvspan(group.index[0], group.index[-1], alpha=0.2, color=color)
        
        ax.set_title('Strategy Performance Comparison')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return fig
        
    def get_best_strategy_for_current_regime(self):
        """
        Determine the best performing strategy for the current regime.
        
        Returns:
        --------
        str
            Name of the best strategy for current regime
        """
        if self.current_regime is None or not self.results:
            return None
            
        regime_data = self.detect_regimes()
        if regime_data is None:
            return None
            
        current_regime_mask = regime_data['regime'] == self.current_regime['state']  # filter for current regime
        current_regime_data = regime_data[current_regime_mask]  # data in current regime periods
        
        regime_performance = {}  # evaluate each strategy in current regime
        for name, result in self.results.items():
            strategy_returns = result['returns']  # get strategy returns
            regime_returns = strategy_returns[current_regime_mask]  # filter returns for regime periods
            
            total_return = (1 + regime_returns).prod() - 1  # calculate total return
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0  # risk-adjusted return
            
            regime_performance[name] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe
            }
        
        best_strategy = max(regime_performance.items(),  # find best strategy based on Sharpe
                           key=lambda x: x[1]['sharpe_ratio'])[0]
        
        return best_strategy
        
    def train_strategy_weights(self, training_data):
        """
        Train a model to predict strategy weights based on market regime.
        
        Parameters:
        -----------
        training_data : dict
            Training data dictionary from Simulator.generate_training_samples()
            
        Returns:
        --------
        dict
            Trained model and performance metrics
        """
        if not training_data or 'feature_matrix' not in training_data:
            print("No valid training data provided.")
            return None
            
        if not self.strategies:
            print("No strategies registered. Register strategies first.")
            return None
            
        try:
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_squared_error, r2_score
            from sklearn.model_selection import train_test_split
            
            # Extract features and target variable
            feature_matrix = training_data['feature_matrix']
            
            # Train models for each strategy to predict performance based on regime
            model_results = {}
            
            for strategy_name in self.strategies.keys():
                print(f"Training model for {strategy_name} strategy...")
                
                # For demonstration, we're using the future_state_num as our target
                # In a real system, you'd use historical strategy performance in each regime
                X = feature_matrix[[col for col in feature_matrix.columns 
                                  if col not in ['date', 'future_state', 'future_state_num', 'future_return']]]
                y = feature_matrix['future_state_num']  # This would normally be strategy performance
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train model (Ridge regression works well for this type of data)
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # For strategy strength on -1 to 1 scale, ensure predictions are in range
                y_pred_clipped = np.clip(y_pred, -1, 1)
                
                model_results[strategy_name] = {
                    'model': model,
                    'mse': mse,
                    'r2': r2,
                    'feature_importance': dict(zip(X.columns, model.coef_))
                }
                
                print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
            
            return model_results
            
        except ImportError as e:
            print(f"Required packages not installed: {e}")
            return None
        except Exception as e:
            print(f"Error training models: {e}")
            return None
            
    def predict_optimal_strategy_weights(self, current_features, trained_models):
        """
        Predict optimal strategy weights based on current market conditions.
        
        Parameters:
        -----------
        current_features : dict
            Dictionary of current market features (normalized to -1 to 1)
        trained_models : dict
            Dictionary of trained models from train_strategy_weights()
            
        Returns:
        --------
        dict
            Strategy weights on -1 to 1 scale
        """
        if not trained_models:
            return None
            
        try:
            import pandas as pd
            
            # Convert features to DataFrame
            features_df = pd.DataFrame([current_features])
            
            # Predict strategy weights
            weights = {}
            for strategy_name, model_data in trained_models.items():
                model = model_data['model']
                prediction = model.predict(features_df)[0]
                
                # Ensure prediction is in -1 to 1 range
                weights[strategy_name] = np.clip(prediction, -1, 1)
            
            # Normalize weights to represent allocation percentages
            total_abs = sum(abs(w) for w in weights.values())
            if total_abs > 0:
                normalized_weights = {k: v / total_abs for k, v in weights.items()}
            else:
                normalized_weights = {k: 0 for k in weights.keys()}
                
            return {
                'raw_weights': weights,  # -1 to 1 scale (negative means short)
                'normalized_weights': normalized_weights  # relative allocations
            }
            
        except Exception as e:
            print(f"Error predicting strategy weights: {e}")
            return None