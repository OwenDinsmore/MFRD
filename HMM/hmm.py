import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------------------------------------------
# Market Regime Detection Using Hidden Markov Models
# This implementation identifies three distinct market regimes (bearish, neutral, bullish)
# based on historical price data. The algorithm leverages return distributions and volatility 
# patterns to determine probabilistic regime states.
#
# Data shapes:
# Input: DataFrame with columns including 'close' (can add 'returns', 'volatility')
# Features: 2D array of [returns, volatility]
# Output: Discrete states (0,1,2) mapped to market regimes with probabilities
#------------------------------------------------------------------------------------------------------------------
class MarketRegimeHMM:
    def __init__(self, n_states=3, n_iter=1000, random_state=42):
        """
        Initialize the Hidden Markov Model.
        
        Parameters:
        -----------
        n_states : int, default=3
            Number of hidden states (typically bearish, neutral, bullish)
        n_iter : int, default=1000
            Number of iterations for model training
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.states = None
        
        # State mapping - this will be updated after fitting
        # Initial mapping will be replaced based on return characteristics
        self.state_labels = {
            0: "Bearish",
            1: "Neutral", 
            2: "Bullish"
        }
        
        # Direct state index to score mapping (-1, 0, 1)
        # This will be updated after fitting to map HMM states properly
        self.state_indices = {
            0: -1,  # Bearish
            1: 0,   # Neutral
            2: 1    # Bullish
        }
        
        self.state_probabilities = None
        self.latest_state = None
        self.latest_probability = None
        
        # Mapping from state labels to scores
        self.state_scores = {
            "Bearish": -1,
            "Neutral": 0,
            "Bullish": 1
        }
        
        # Debug information
        self.debug_info = {
            'state_returns': {},     # Average returns in each state
            'state_volatility': {},  # Average volatility in each state
            'transition_matrix': None,  # State transition probabilities
            'emission_means': None,  # Emission distribution means
            'emission_covars': None  # Emission distribution covariances
        }
    
    #------------------------------------------------------------------------------------------------------------------
    # Feature engineering for HMM input
    #------------------------------------------------------------------------------------------------------------------
    def _prepare_features(self, data):
        """
        Prepare features for the HMM model.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with at least 'close' column
            
        Returns:
        --------
        features : numpy.ndarray
            Scaled features for HMM training
        """
        if 'returns' not in data.columns:
            data.loc[:, 'returns'] = np.log(data['close'] / data['close'].shift(1))  # calculate log returns
        
        if 'volatility' not in data.columns:
            data.loc[:, 'volatility'] = data['returns'].rolling(window=20).std()  # 20-day rolling volatility
        
        data = data.dropna()  # remove NaN values
        
        features = data[['returns', 'volatility']].values  # select features
        scaled_features = self.scaler.fit_transform(features)  # standardize features
        
        return scaled_features
    
    #------------------------------------------------------------------------------------------------------------------
    # Model training and state labeling
    #------------------------------------------------------------------------------------------------------------------
    def fit(self, data):
        """
        Fit the HMM model to historical price data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with at least 'close' column
            
        Returns:
        --------
        self : object
            Returns self
        """
        features = self._prepare_features(data)
        print(f"Fitting HMM model with {self.n_states} states on {features.shape[0]} data points...")
        
        # fit le model
        self.model.fit(features)
        
        # store params (just debugging ;) )
        self.debug_info['transition_matrix'] = self.model.transmat_
        self.debug_info['emission_means'] = self.model.means_
        self.debug_info['emission_covars'] = self.model.covars_
        
        print("HMM model fitted. Predicting states...")
        self.states = self.model.predict(features)
        self.state_probabilities = self.model.predict_proba(features)
        self.latest_state = self.states[-1]
        self.latest_probability = self.state_probabilities[-1][self.latest_state]
        
        # state characteristics for labeling
        data_with_states = data.copy().dropna()
        data_with_states['state'] = self.states
        
        # calc average returns and volatility for each state
        state_returns = data_with_states.groupby('state')['returns'].mean()
        state_volatility = data_with_states.groupby('state')['returns'].std()
        state_counts = data_with_states.groupby('state').size()
        
        # store for debugging YAY
        self.debug_info['state_returns'] = state_returns.to_dict()
        self.debug_info['state_volatility'] = state_volatility.to_dict()
        self.debug_info['state_counts'] = state_counts.to_dict()
        
        print("\nState Analysis:")
        for state in range(self.n_states):
            if state in state_returns:
                print(f"State {state}: Avg Return={state_returns[state]:.6f}, "
                      f"Volatility={state_volatility[state]:.6f}, "
                      f"Count={state_counts[state]} ({state_counts[state]/len(data_with_states)*100:.1f}%)")
        
        # order states by mean returns for labeling
        state_means = state_returns.sort_values()
        states_ordered = state_means.index.tolist()
        
        # re-map the state labels based on returns
        self.state_labels = {}
        if len(states_ordered) == 3:
            self.state_labels = {
                states_ordered[0]: "Bearish",  # lowest returns
                states_ordered[1]: "Neutral",  # middle returns
                states_ordered[2]: "Bullish"   # highest returns
            }
        else:
            for i, state in enumerate(states_ordered):
                if i == 0:
                    self.state_labels[state] = "Bearish"  # lowest returns
                elif i == len(states_ordered) - 1:
                    self.state_labels[state] = "Bullish"  # highest returns
                else:
                    self.state_labels[state] = "Neutral"  # middle returns
        
        # Create direct state-to-score mapping (-1, 0, 1)
        self.state_indices = {}
        for state, label in self.state_labels.items():
            if label == "Bearish":
                self.state_indices[state] = -1
            elif label == "Neutral":
                self.state_indices[state] = 0
            elif label == "Bullish":
                self.state_indices[state] = 1
        
        print("\nState to Regime Mapping:")
        for state, label in self.state_labels.items():
            score = self.state_indices[state]
            print(f"State {state} -> {label} (Score: {score})")
            
        # Print transition matrix
        print("\nState Transition Probabilities:")
        print("From\\To ", end="")
        for i in range(self.n_states):
            print(f"{self.state_labels[i]:8s} ", end="")
        print()
        
        for i in range(self.n_states):
            print(f"{self.state_labels[i]:8s} ", end="")
            for j in range(self.n_states):
                print(f"{self.model.transmat_[i, j]:.6f} ", end="")
            print()
            
        return self
    
    #------------------------------------------------------------------------------------------------------------------
    # State prediction 
    #------------------------------------------------------------------------------------------------------------------
    def predict(self, data):
        """
        Predict market regimes for the given data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with at least 'close' column
            
        Returns:
        --------
        states : numpy.ndarray
            Predicted regime states
        """
        features = self._prepare_features(data)
        states = self.model.predict(features)
        return states
    
    #------------------------------------------------------------------------------------------------------------------
    # Probability estimation
    #------------------------------------------------------------------------------------------------------------------
    def predict_proba(self, data):
        """
        Predict market regime probabilities for the given data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with at least 'close' column
            
        Returns:
        --------
        probabilities : numpy.ndarray
            Predicted regime probabilities
        """
        features = self._prepare_features(data)
        probabilities = self.model.predict_proba(features)
        return probabilities
    
    #------------------------------------------------------------------------------------------------------------------
    # Current market state assessment
    #------------------------------------------------------------------------------------------------------------------
    def get_current_regime(self):
        """
        Get the current market regime and its probability.
        
        Returns:
        --------
        regime : dict
            Dictionary with 'state', 'probability', 'score', and 'raw_state' keys
        """
        if self.latest_state is None:
            return None
        
        state_label = self.state_labels[self.latest_state]
        direct_score = self.state_indices[self.latest_state]  # Direct -1, 0, 1 mapping
        
        # Also calculate weighted score based on all state probabilities
        weighted_score = self.get_regime_score(self.state_probabilities[-1])
        
        return {
            'state': state_label,
            'probability': self.latest_probability,
            'score': direct_score,  # Direct -1, 0, 1 score
            'weighted_score': weighted_score,  # Probability-weighted score (-1 to 1)
            'raw_state': self.latest_state,  # The raw state number from HMM
            'all_probabilities': {self.state_labels[i]: self.state_probabilities[-1][i] 
                                 for i in range(self.n_states)}  # All state probabilities
        }
        
    def get_regime_score(self, state_probabilities=None):
        """
        Calculate a weighted regime score on a -1 to 1 scale.
        
        Parameters:
        -----------
        state_probabilities : numpy.ndarray, optional
            State probabilities to use. Defaults to latest probabilities.
            
        Returns:
        --------
        float
            Weighted regime score from -1 (bearish) to 1 (bullish)
        """
        if state_probabilities is None:
            if self.state_probabilities is None or len(self.state_probabilities) == 0:
                return 0
            state_probabilities = self.state_probabilities[-1]
        
        score = 0
        for state, prob in enumerate(state_probabilities):
            if state in self.state_indices:
                # Directly use the -1, 0, 1 mapping
                state_score = self.state_indices[state]
                score += prob * state_score
        
        return score
    
    #------------------------------------------------------------------------------------------------------------------
    # Visualization
    #------------------------------------------------------------------------------------------------------------------
    def evaluate_accuracy(self, data, future_window=5, threshold=0.02):
        """
        Evaluate the model's prediction accuracy by comparing regime predictions
        to future returns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data with state predictions
        future_window : int, default=5
            Number of days to look ahead for return calculation
        threshold : float, default=0.02
            Threshold for determining good/bad performance
            
        Returns:
        --------
        dict
            Dictionary with accuracy metrics
        """
        if self.states is None:
            print("Model not trained yet. Call fit() first.")
            return None
        
        # Create clean data copy with state predictions
        eval_data = data.copy().dropna()
        eval_data = eval_data.iloc[:(len(self.states))]
        eval_data['state'] = self.states
        eval_data['state_score'] = eval_data['state'].map(self.state_indices)
        
        # Calculate future returns
        eval_data['future_return'] = eval_data['close'].pct_change(future_window).shift(-future_window)
        
        # Label future performance
        eval_data['future_state'] = 'neutral'
        eval_data.loc[eval_data['future_return'] > threshold, 'future_state'] = 'good'
        eval_data.loc[eval_data['future_return'] < -threshold, 'future_state'] = 'bad'
        
        # Map to numeric scale
        future_state_map = {'bad': -1, 'neutral': 0, 'good': 1}
        eval_data['future_state_num'] = eval_data['future_state'].map(future_state_map)
        
        # Remove NaN values
        eval_data = eval_data.dropna(subset=['future_return', 'future_state_num'])
        
        # calc correlation between regime score and future returns
        correlation = eval_data[['state_score', 'future_return']].corr().iloc[0, 1]
        
        # calculate directional accuracy
        matches = (eval_data['state_score'] * eval_data['future_state_num'] >= 0).sum()
        total = len(eval_data)
        directional_accuracy = matches / total if total > 0 else 0
        
        # calculate accuracy by regime
        accuracy_by_regime = {}
        for regime in ['Bearish', 'Neutral', 'Bullish']:
            # find states corresponding to this regime
            regime_states = [s for s, label in self.state_labels.items() if label == regime]
            
            if regime_states:
                # Filter data for this regime
                regime_mask = eval_data['state'].isin(regime_states)
                regime_data = eval_data[regime_mask]
                
                if len(regime_data) > 0:
                    # Calculate performance metrics
                    avg_future_return = regime_data['future_return'].mean()
                    
                    # Count future states
                    future_counts = regime_data['future_state'].value_counts()
                    future_good = future_counts.get('good', 0)
                    future_neutral = future_counts.get('neutral', 0)
                    future_bad = future_counts.get('bad', 0)
                    
                    total_regime = len(regime_data)
                    pct_good = future_good / total_regime * 100 if total_regime > 0 else 0
                    pct_neutral = future_neutral / total_regime * 100 if total_regime > 0 else 0
                    pct_bad = future_bad / total_regime * 100 if total_regime > 0 else 0
                    
                    # expected direction based on regime
                    expected_direction = None
                    if regime == 'Bearish':
                        expected_direction = 'bad'
                    elif regime == 'Neutral':
                        expected_direction = 'neutral'
                    elif regime == 'Bullish':
                        expected_direction = 'good'
                    
                    expected_count = future_counts.get(expected_direction, 0)
                    directional_accuracy = expected_count / total_regime if total_regime > 0 else 0
                    
                    accuracy_by_regime[regime] = {
                        'count': total_regime,
                        'avg_future_return': avg_future_return,
                        'pct_good': pct_good,
                        'pct_neutral': pct_neutral,
                        'pct_bad': pct_bad,
                        'directional_accuracy': directional_accuracy
                    }
        
        # calc transition stats
        state_transitions = []
        for i in range(1, len(eval_data)):
            prev_state = eval_data['state'].iloc[i-1]
            curr_state = eval_data['state'].iloc[i]
            if prev_state != curr_state:
                transition = (prev_state, curr_state)
                state_transitions.append(transition)
        
        # return metrics
        results = {
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'accuracy_by_regime': accuracy_by_regime,
            'state_transitions': len(state_transitions),
            'avg_transition_freq': len(eval_data) / (len(state_transitions) + 1) if state_transitions else None
        }
        
        # pring summary for debug
        print("\nHMM Model Evaluation:")
        print(f"Total samples: {total}")
        print(f"Correlation between regime score and future returns: {correlation:.4f}")
        print(f"Overall directional accuracy: {directional_accuracy:.4f} ({matches} of {total})")
        
        print("\nAccuracy by regime:")
        for regime, metrics in accuracy_by_regime.items():
            print(f"- {regime} regime ({metrics['count']} samples):")
            print(f"  * Avg future return: {metrics['avg_future_return']:.4f}")
            print(f"  * Next state distribution: Good={metrics['pct_good']:.1f}%, Neutral={metrics['pct_neutral']:.1f}%, Bad={metrics['pct_bad']:.1f}%")
            print(f"  * Directional accuracy: {metrics['directional_accuracy']:.4f}")
        
        print(f"\nState transitions: {len(state_transitions)}")
        print(f"Average samples between transitions: {results['avg_transition_freq']:.1f}")
        
        return results

    def plot_regimes(self, data, price_column='close'):
        """
        Plot the price data with colored backgrounds for different regimes.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Historical price data
        price_column : str, default='close'
            Column name for price data to plot
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if self.states is None:
            print("Model not trained yet. Call fit() first.")
            return None
        
        plot_data = data.copy().dropna()  # create clean data copy
        plot_data = plot_data.iloc[:(len(self.states))]  # match length to states
        plot_data['state'] = self.states  # add state column
        plot_data['state_score'] = plot_data['state'].map(self.state_indices)  # add -1,0,1 score
        
        # create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # plot price data with colored backgrounds
        ax1.plot(plot_data.index, plot_data[price_column], color='black', lw=1.5)  # plot price data
        
        for state in range(self.n_states):
            mask = plot_data['state'] == state  # create mask for current state
            color = 'red' if self.state_labels[state] == 'Bearish' else \
                    'green' if self.state_labels[state] == 'Bullish' else 'yellow'  # set colors
            alpha = 0.2  # transparency
            
            if mask.any():
                ax1.fill_between(
                    plot_data.index, 
                    plot_data[price_column].min(), 
                    plot_data[price_column].max(),
                    where=mask, 
                    color=color, 
                    alpha=alpha, 
                    label=f"{self.state_labels[state]} (State {state})"
                )  # create colored background
        
        ax1.set_title('Market Regimes Detected by HMM')
        ax1.set_ylabel(price_column)
        ax1.legend(loc='upper left')
        
        # Plot regime score on a separate subplot
        ax2.plot(plot_data.index, plot_data['state_score'], color='blue', lw=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax2.fill_between(plot_data.index, plot_data['state_score'], 0, 
                        where=plot_data['state_score'] >= 0, color='green', alpha=0.3)
        ax2.fill_between(plot_data.index, plot_data['state_score'], 0, 
                        where=plot_data['state_score'] < 0, color='red', alpha=0.3)
        ax2.set_ylabel('Regime Score (-1 to 1)')
        ax2.set_xlabel('Date')
        ax2.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        
        return fig


#------------------------------------------------------------------------------------------------------------------
# Example usage
#------------------------------------------------------------------------------------------------------------------
def demo():
    try:
        import yfinance as yf
        ticker = "SPY"  # s&p 500 etf
        start_date = "2018-01-01"
        end_date = "2023-01-01"
        data = yf.download(ticker, start=start_date, end=end_date)  # get market data
        
        regime_hmm = MarketRegimeHMM(n_states=3)  # create model
        regime_hmm.fit(data)  # train model
        
        current_regime = regime_hmm.get_current_regime()  # get regime
        print(f"Current market regime: {current_regime['state']} (Probability: {current_regime['probability']:.2f})")
        
        fig = regime_hmm.plot_regimes(data)  # visualize
        plt.show()
        
    except ImportError:
        print("yfinance not installed. Install with: pip install yfinance")
    except Exception as e:
        print(f"Error in demo: {e}")


if __name__ == "__main__":
    demo()