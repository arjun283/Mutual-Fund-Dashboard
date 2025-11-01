import unittest
import pandas as pd
import numpy as np
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import (
    rank_funds,
    suggest_allocation,
    predict_returns,
    run_monte_carlo_simulation
)

class TestModelFunctions(unittest.TestCase):
    def setUp(self):
        # Create sample metrics data for multiple funds
        self.metrics_data = pd.DataFrame({
            'Ticker': ['FUND1', 'FUND2', 'FUND3', 'FUND4', 'FUND5'],
            'Fund Name': ['Fund 1', 'Fund 2', 'Fund 3', 'Fund 4', 'Fund 5'],
            'Annualized Return (%)': [12.5, 10.2, 15.7, 8.3, 11.9],
            'Volatility (%)': [18.2, 12.5, 22.1, 10.8, 15.3],
            'Sharpe Ratio': [0.52, 0.58, 0.62, 0.49, 0.59],
            'Sortino Ratio': [0.78, 0.85, 0.91, 0.72, 0.88],
            'Max Drawdown (%)': [25.3, 18.7, 30.2, 15.5, 22.1],
            'Beta': [1.2, 0.85, 1.35, 0.75, 1.05],
            'Alpha': [2.5, 1.8, 3.2, 1.2, 2.1],
            '3M Momentum': [3.2, 2.5, 4.1, 1.8, 2.9],
            '6M Momentum': [7.5, 5.8, 9.2, 4.2, 6.5],
            '12M Momentum': [15.3, 12.1, 18.5, 9.7, 13.8],
            'Liquidity Score': [0.85, 0.92, 0.78, 0.95, 0.88],
            'Market Sentiment': [0.6, 0.7, 0.5, 0.8, 0.65]
        })
        
        # Create sample historical returns for Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        self.returns_data = pd.DataFrame(
            np.random.normal(0.01, 0.05, size=(100, 5)),
            columns=['FUND1', 'FUND2', 'FUND3', 'FUND4', 'FUND5']
        )
    
    def test_rank_funds_low_risk(self):
        """Test fund ranking with low risk tolerance"""
        ranked_funds = rank_funds(self.metrics_data, risk_tolerance='low')
        
        # Check that all funds are ranked
        self.assertEqual(len(ranked_funds), len(self.metrics_data))
        
        # Check that the DataFrame has a 'Composite Score' column
        self.assertIn('Composite Score', ranked_funds.columns)
        
        # For low risk, funds with lower volatility, drawdown, and beta should rank higher
        # FUND4 has the lowest risk metrics and should be ranked higher
        top_fund = ranked_funds.iloc[0]['Ticker']
        self.assertEqual(top_fund, 'FUND4')
    
    def test_rank_funds_medium_risk(self):
        """Test fund ranking with medium risk tolerance"""
        ranked_funds = rank_funds(self.metrics_data, risk_tolerance='medium')
        
        # Check that all funds are ranked
        self.assertEqual(len(ranked_funds), len(self.metrics_data))
        
        # For medium risk, there should be a balance between return and risk
        # FUND2 or FUND5 should be ranked higher due to balanced metrics
        top_fund = ranked_funds.iloc[0]['Ticker']
        self.assertIn(top_fund, ['FUND2', 'FUND5'])
    
    def test_rank_funds_high_risk(self):
        """Test fund ranking with high risk tolerance"""
        ranked_funds = rank_funds(self.metrics_data, risk_tolerance='high')
        
        # Check that all funds are ranked
        self.assertEqual(len(ranked_funds), len(self.metrics_data))
        
        # For high risk, funds with higher returns and momentum should rank higher
        # FUND3 has the highest return and momentum and should be ranked higher
        top_fund = ranked_funds.iloc[0]['Ticker']
        self.assertEqual(top_fund, 'FUND3')
    
    def test_suggest_allocation(self):
        """Test allocation suggestion"""
        # Rank funds first
        ranked_funds = rank_funds(self.metrics_data, risk_tolerance='medium')
        
        # Test allocation for different number of funds
        for top_n in [1, 3, 5]:
            invest_amount = 100000
            allocation = suggest_allocation(ranked_funds, top_n, invest_amount)
            
            # Check that the allocation has the right number of funds
            self.assertEqual(len(allocation), top_n)
            
            # Check that the allocation sums to the investment amount
            self.assertAlmostEqual(allocation['Amount'].sum(), invest_amount, places=2)
            
            # Check that the allocation is proportional to the composite score
            if top_n > 1:
                # Higher ranked funds should have higher allocation
                self.assertTrue(allocation['Amount'].iloc[0] > allocation['Amount'].iloc[-1])
    
    def test_predict_returns(self):
        """Test return prediction"""
        # Create features and target for prediction
        X = self.metrics_data[['Volatility (%)', 'Sharpe Ratio', 'Beta', 'Alpha', '3M Momentum', '6M Momentum']]
        y = self.metrics_data['Annualized Return (%)']
        
        # Test prediction with different models
        for model_type in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            try:
                predictions = predict_returns(X, y, X, model_type=model_type)
                
                # Check that predictions have the right shape
                self.assertEqual(len(predictions), len(X))
                
                # Check that predictions are reasonable (positive for this test data)
                self.assertTrue(all(predictions >= 0))
            except ImportError:
                # Skip test if the model type is not available
                print(f"Skipping {model_type} test due to missing dependencies")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        # Create allocation
        allocation = pd.DataFrame({
            'Ticker': ['FUND1', 'FUND2', 'FUND3'],
            'Weight': [0.5, 0.3, 0.2]
        })
        
        # Run simulation
        initial_investment = 100000
        time_horizon = 3
        simulation_results = run_monte_carlo_simulation(
            self.returns_data, allocation, initial_investment, time_horizon, num_simulations=100
        )
        
        # Check that the simulation results contain the expected keys
        expected_keys = ['simulations', 'percentiles', 'mean_final_value', 'median_final_value', 'min_final_value', 'max_final_value']
        for key in expected_keys:
            self.assertIn(key, simulation_results)
        
        # Check that the simulations have the right shape
        expected_periods = time_horizon * 12 + 1  # Monthly periods plus initial
        self.assertEqual(simulation_results['simulations'].shape, (100, expected_periods))
        
        # Check that the initial value is correct
        self.assertEqual(simulation_results['simulations'].iloc[:, 0].mean(), initial_investment)
        
        # Check that the percentiles are in ascending order
        percentiles = simulation_results['percentiles']
        self.assertTrue(percentiles[0] <= percentiles[1] <= percentiles[2])

    if __name__ == '__main__':
        unittest.main()