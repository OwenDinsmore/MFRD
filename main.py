import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from Model.model import Model
from Simulator.simulator import Simulator
from Strategies.momentum_strategy import MomentumStrategy
from Strategies.mean_reversion_strategy import MeanReversionStrategy
from Strategies.fama_french_strategy import FamaFrenchStrategy
from Strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy


#------------------------------------------------------------------------------------------------------------------
# MFRD System Demo
# Main entry point for demonstrating the Multi-Factor Regime Detection system
# Loads data, runs regime detection, registers strategies and runs backtests
#------------------------------------------------------------------------------------------------------------------
def main():
    """
    Main function to demonstrate the MFRD system.
    """
    print("Multi-Factor Regime Detection (MFRD) System")
    print("-------------------------------------------")
    
    # Initialize model
    model = Model()
    
    # Load sample data
    print("\nLoading market data...")
    model.load_data("SPY", "2018-01-01", "2023-01-01")
    
    # Detect market regimes
    print("\nDetecting market regimes...")
    regime_data = model.detect_regimes()
    
    # Print regime distribution
    if regime_data is not None:
        regime_counts = regime_data['regime'].value_counts()
        print("\nRegime distribution:")
        for regime, count in regime_counts.items():
            percentage = count / len(regime_data) * 100
            print(f" - {regime}: {count} days ({percentage:.1f}%)")
    
    # Register strategies
    print("\nRegistering strategies...")
    model.register_strategy("Momentum", MomentumStrategy)
    model.register_strategy("Mean Reversion", MeanReversionStrategy)
    model.register_strategy("Fama-French", FamaFrenchStrategy)
    model.register_strategy("Regime Adaptive", RegimeAdaptiveStrategy)
    
    # Backtest strategies
    print("\nRunning backtest...")
    results = model.backtest_all_strategies()
    
    # Print performance summary
    print("\nStrategy Performance:")
    for name, result in results.items():
        print(f"\n{name} Strategy:")
        print(f" - Total Return: {result['total_return']*100:.2f}%")
        print(f" - Annual Return: {result['annual_return']*100:.2f}%")
        print(f" - Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f" - Max Drawdown: {result['max_drawdown']*100:.2f}%")
        print(f" - Win Rate: {result['win_rate']*100:.2f}%")
    
    # Plot strategy performance
    print("\nPlotting performance comparison...")
    fig = model.plot_strategy_performance(benchmark="SPY")
    plt.savefig("strategy_performance.png")
    print("Plot saved as 'strategy_performance.png'")
    
    # Show best strategy for current regime
    if model.current_regime:
        best_strategy = model.get_best_strategy_for_current_regime()
        print(f"\nCurrent Market Regime: {model.current_regime['state']} (Probability: {model.current_regime['probability']:.2f})")
        print(f"Best Strategy for Current Regime: {best_strategy}")
    
    print("\nMFRD demonstration completed!")

#------------------------------------------------------------------------------------------------------------------
# Data Generation and Model Training
# This function demonstrates the two-layer approach to model training:
# 1. HMM for market regime detection (first layer)
# 2. Strategy weight optimization based on regime probabilities (second layer)
#------------------------------------------------------------------------------------------------------------------
def train_two_layer_model():
    """
    Generate labeled training data and train the two-layer model system.
    """
    print("Multi-Factor Regime Detection (MFRD) - Two-Layer Model Training")
    print("------------------------------------------------------------")
    
    # Initialize simulator and model
    simulator = Simulator()
    model = Model()
    
    # Define parameters
    market_ticker = 'SPY'
    additional_tickers = ['QQQ', 'IWM', 'EFA', 'TLT']  # ETFs for different market segments
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    future_window = 5  # 5-day future window for labeling
    threshold = 0.02  # 2% threshold for good/bad performance
    n_samples = 100  # Generate 100 random samples
    
    print("\n1. Layer 1: Market Regime Detection (HMM)")
    print("----------------------------------------")
    
    # Train the HMM model for regime detection
    print("Training HMM model on market data...")
    model.load_data(market_ticker, start_date, end_date)
    regime_data = model.detect_regimes()
    
    # Print regime distribution with scores
    if regime_data is not None:
        regime_counts = regime_data['regime'].value_counts()
        print("\nRegime distribution:")
        for regime, count in regime_counts.items():
            percentage = count / len(regime_data) * 100
            score_avg = regime_data.loc[regime_data['regime'] == regime, 'regime_score'].mean()
            print(f" - {regime}: {count} days ({percentage:.1f}%), Avg Score: {score_avg:.2f}")
    
    print("\n2. Layer 2: Strategy Weight Optimization")
    print("--------------------------------------")
    
    # Register strategies
    print("Registering strategies...")
    model.register_strategy("Momentum", MomentumStrategy)
    model.register_strategy("Mean Reversion", MeanReversionStrategy)
    model.register_strategy("Fama-French", FamaFrenchStrategy)
    model.register_strategy("Regime Adaptive", RegimeAdaptiveStrategy)
    
    # Generate training data with normalized features (all on -1 to 1 scale)
    print("\nGenerating normalized training data...")
    training_data = simulator.generate_training_samples(
        market_ticker=market_ticker,
        additional_tickers=additional_tickers,
        start_date=start_date,
        end_date=end_date,
        future_window=future_window,
        threshold=threshold,
        n_samples=n_samples
    )
    
    if training_data and 'feature_matrix' in training_data:
        feature_matrix = training_data['feature_matrix']
        
        # Display sample counts
        if 'samples' in training_data:
            samples = training_data['samples']
            state_counts = samples['future_state'].value_counts()
            print("\nFuture state distribution in samples:")
            for state, count in state_counts.items():
                percentage = count / len(samples) * 100
                avg_score = samples.loc[samples['future_state'] == state, 'future_state_num'].mean()
                print(f" - {state}: {count} samples ({percentage:.1f}%), Avg Score: {avg_score:.2f}")
        
        # Train strategy weight optimization models
        print("\nTraining strategy weight optimization models...")
        strategy_models = model.train_strategy_weights(training_data)
        
        if strategy_models:
            # Save feature matrix for reference
            feature_matrix.to_csv('training_features.csv')
            print("\nSaved feature matrix to 'training_features.csv'")
            
            # Test with current market conditions
            print("\nPredicting optimal strategy weights for current conditions...")
            
            # Get current features
            if model.current_regime:
                current_features = {
                    'market_regime_score': model.current_regime.get('score', 0),
                    'market_returns': 0,  # Would be calculated from current data
                    'market_volatility': 0,  # Would be calculated from current data
                    'market_rsi': 0,  # Would be calculated from current data
                    'market_price_trend': 0,  # Would be calculated from current data
                    'market_trend_strength': 0  # Would be calculated from current data
                }
                
                # Predict strategy weights
                weights = model.predict_optimal_strategy_weights(current_features, strategy_models)
                
                if weights:
                    print("\nOptimal strategy weights (raw scores on -1 to 1 scale):")
                    for strategy, weight in weights['raw_weights'].items():
                        print(f" - {strategy}: {weight:.4f}")
                    
                    print("\nNormalized portfolio allocation:")
                    for strategy, weight in weights['normalized_weights'].items():
                        allocation = weight * 100
                        direction = "Long" if weight > 0 else "Short"
                        print(f" - {strategy}: {direction} {abs(allocation):.2f}%")
    
    print("\nTwo-layer model training completed!")

#------------------------------------------------------------------------------------------------------------------
# Production Demo
# This function demonstrates how to use the MFRD system in production
#------------------------------------------------------------------------------------------------------------------
def run_production_demo():
    """
    Demonstrate how to use the MFRD system in production.
    """
    print("Multi-Factor Regime Detection (MFRD) - Production Demo")
    print("---------------------------------------------------")
    
    # Initialize simulator and model
    simulator = Simulator()
    model = Model()
    
    # Set market parameters
    market_ticker = 'SPY'
    additional_tickers = ['QQQ', 'IWM', 'EFA', 'TLT']
    
    # Step 1: Train the models (or load pre-trained models)
    print("\n1. Training Models (would typically be pre-loaded in production)")
    print("-----------------------------------------------------------")
    
    # Initialize model
    model.load_data(market_ticker, "2018-01-01", "2023-01-01")
    
    # Train HMM model for regime detection
    regime_data = model.detect_regimes()
    print(f"HMM model trained: {model.current_regime['state']} regime detected")
    
    # Register strategies
    model.register_strategy("Momentum", MomentumStrategy)
    model.register_strategy("Mean Reversion", MeanReversionStrategy)
    model.register_strategy("Fama-French", FamaFrenchStrategy)
    model.register_strategy("Regime Adaptive", RegimeAdaptiveStrategy)
    
    # Step 2: Get current market features (normalized to -1 to 1 scale)
    print("\n2. Getting Current Market Features")
    print("--------------------------------")
    
    current_features = simulator.get_current_market_features(
        market_ticker=market_ticker,
        additional_tickers=additional_tickers,
        lookback_days=252
    )
    
    if current_features:
        print("\nCurrent Market Features (normalized to -1 to 1 scale):")
        for feature, value in current_features.items():
            if feature != 'date':
                print(f" - {feature}: {value:.4f}")
        
        # Add regime score from HMM
        current_features['market_regime_score'] = model.current_regime.get('score', 0)
        print(f" - market_regime_score: {current_features['market_regime_score']:.4f} ({model.current_regime['state']})")
        
        # Step A: Simple approach - Select best strategy for current regime
        best_strategy = model.get_best_strategy_for_current_regime()
        print(f"\nSimple approach: Best strategy for current regime: {best_strategy}")
        
        # Step B: Advanced approach - Use models to predict optimal strategy weights
        # Train strategy weight optimization models (in production, these would be pre-trained)
        # For demonstration, we're using a placeholder model
        from sklearn.linear_model import Ridge
        
        # Create simple placeholder model that outputs weights based on regime score
        class PlaceholderModel:
            def predict(self, X):
                regime_score = X.iloc[0].get('market_regime_score', 0)
                
                # Simple rules for demonstration:
                # Bearish regime: favor Mean Reversion
                # Neutral regime: favor Fama-French
                # Bullish regime: favor Momentum
                result = np.zeros(1)
                if regime_score < -0.3:  # Bearish
                    result[0] = -0.8
                elif regime_score > 0.3:  # Bullish
                    result[0] = 0.8
                else:  # Neutral
                    result[0] = 0.2
                return result
        
        # Create placeholder models
        strategy_models = {
            "Momentum": {"model": PlaceholderModel(), "feature_importance": {"market_regime_score": 0.8}},
            "Mean Reversion": {"model": PlaceholderModel(), "feature_importance": {"market_regime_score": -0.8}},
            "Fama-French": {"model": PlaceholderModel(), "feature_importance": {"market_regime_score": 0.2}},
            "Regime Adaptive": {"model": PlaceholderModel(), "feature_importance": {"market_regime_score": 0.5}}
        }
        
        print("\nAdvanced approach: Predicting optimal strategy weights...")
        weights = model.predict_optimal_strategy_weights(current_features, strategy_models)
        
        if weights:
            print("\nStrategy signals (-1 to 1 scale):")
            for strategy, weight in weights['raw_weights'].items():
                signal = "Neutral"
                if weight > 0.3:
                    signal = "Long"
                elif weight < -0.3:
                    signal = "Short"
                print(f" - {strategy}: {weight:.4f} ({signal})")
            
            print("\nPortfolio allocation:")
            for strategy, weight in weights['normalized_weights'].items():
                allocation = weight * 100
                direction = "Long" if weight > 0 else "Short"
                print(f" - {strategy}: {direction} {abs(allocation):.2f}%")
    
    print("\nProduction demo completed!")

if __name__ == "__main__":
    # Uncomment the function you want to run
    main()  # Run main MFRD system demo
    # train_two_layer_model()  # Run two-layer model training and testing
    # run_production_demo()  # Run production demo