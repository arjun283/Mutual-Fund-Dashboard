import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import (
    calculate_returns,
    calculate_annualized_return,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_beta_alpha,
    calculate_momentum
)

class TestMetricsCalculation(unittest.TestCase):
    def setUp(self):
        # Create sample fund data
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')  # Business days
        n_days = len(dates)
        
        # Create a sample fund with some realistic price movements
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.0005, 0.01, n_days)  # Mean 0.05% daily, std 1%
        
        # Convert returns to prices
        prices = 100 * (1 + daily_returns).cumprod()
        
        # Create DataFrame
        self.fund_data = pd.DataFrame({
            'date': dates,
            'nav': prices
        })
        
        # Create sample benchmark data with different characteristics
        benchmark_returns = np.random.normal(0.0004, 0.008, n_days)  # Lower vol than fund
        benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
        
        self.benchmark_data = pd.DataFrame({
            'date': dates,
            'nav': benchmark_prices
        })
        
        # Calculate returns for testing
        self.returns = calculate_returns(self.fund_data)
        self.benchmark_returns = calculate_returns(self.benchmark_data)
    
    def test_calculate_returns(self):
        """Test that returns are calculated correctly"""
        returns = calculate_returns(self.fund_data)
        
        # Check that returns have the right shape
        self.assertEqual(len(returns), len(self.fund_data) - 1)
        
        # Check that returns are calculated correctly for a few samples
        for i in range(1, 5):
            expected_return = self.fund_data['nav'].iloc[i] / self.fund_data['nav'].iloc[i-1] - 1
            self.assertAlmostEqual(returns.iloc[i-1], expected_return)
    
    def test_calculate_annualized_return(self):
        """Test annualized return calculation"""
        # Calculate annualized return
        ann_return = calculate_annualized_return(self.returns)
        
        # Calculate expected annualized return manually
        total_return = self.fund_data['nav'].iloc[-1] / self.fund_data['nav'].iloc[0] - 1
        years = (self.fund_data['date'].iloc[-1] - self.fund_data['date'].iloc[0]).days / 365.25
        expected_ann_return = (1 + total_return) ** (1 / years) - 1
        
        # Check that the calculated value is close to the expected value
        self.assertAlmostEqual(ann_return, expected_ann_return, places=4)
    
    def test_calculate_volatility(self):
        """Test volatility calculation"""
        # Calculate volatility
        vol = calculate_volatility(self.returns)
        
        # Calculate expected volatility manually
        expected_vol = self.returns.std() * np.sqrt(252)  # Annualize
        
        # Check that the calculated value is close to the expected value
        self.assertAlmostEqual(vol, expected_vol, places=4)
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Calculate Sharpe ratio
        ann_return = calculate_annualized_return(self.returns)
        vol = calculate_volatility(self.returns)
        sharpe = calculate_sharpe_ratio(ann_return, vol, risk_free_rate=0.03)
        
        # Calculate expected Sharpe ratio manually
        expected_sharpe = (ann_return - 0.03) / vol
        
        # Check that the calculated value is close to the expected value
        self.assertAlmostEqual(sharpe, expected_sharpe, places=4)
    
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        # Calculate Sortino ratio
        ann_return = calculate_annualized_return(self.returns)
        sortino = calculate_sortino_ratio(self.returns, ann_return, risk_free_rate=0.03)
        
        # Calculate expected Sortino ratio manually
        downside_returns = self.returns[self.returns < 0]
        downside_deviation = np.sqrt(252) * np.sqrt(np.mean(downside_returns**2))
        expected_sortino = (ann_return - 0.03) / downside_deviation if downside_deviation != 0 else float('inf')
        
        # Check that the calculated value is close to the expected value
        if np.isfinite(expected_sortino):
            self.assertAlmostEqual(sortino, expected_sortino, places=4)
        else:
            self.assertTrue(np.isfinite(sortino))  # Ensure we don't return infinity
    
    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Calculate max drawdown
        max_dd = calculate_max_drawdown(self.returns)
        
        # Calculate expected max drawdown manually
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        expected_max_dd = drawdowns.min()
        
        # Check that the calculated value is close to the expected value
        self.assertAlmostEqual(max_dd, expected_max_dd, places=4)
    
    def test_calculate_beta_alpha(self):
        """Test beta and alpha calculation"""
        # Calculate beta and alpha
        start_date = self.fund_data['date'].min()
        end_date = self.fund_data['date'].max()
        beta, alpha = calculate_beta_alpha(self.returns, self.benchmark_returns, start_date, end_date)
        
        # Calculate expected beta and alpha manually using linear regression
        # Beta is the slope, Alpha is the intercept * 252 (annualized)
        X = self.benchmark_returns.values.reshape(-1, 1)
        y = self.returns.values
        
        # Add a constant to X for the intercept term
        X_with_const = np.column_stack([np.ones(X.shape[0]), X])
        
        # Solve for the coefficients using the normal equation
        coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        
        expected_alpha = coeffs[0] * 252  # Annualize
        expected_beta = coeffs[1]
        
        # Check that the calculated values are close to the expected values
        self.assertAlmostEqual(beta, expected_beta, places=4)
        self.assertAlmostEqual(alpha, expected_alpha, places=4)
    
    def test_calculate_momentum(self):
        """Test momentum calculation"""
        # Calculate momentum
        end_date = self.fund_data['date'].max()
        momentum_3m = calculate_momentum(self.fund_data, end_date, months=3)
        momentum_6m = calculate_momentum(self.fund_data, end_date, months=6)
        momentum_12m = calculate_momentum(self.fund_data, end_date, months=12)
        
        # Calculate expected momentum manually
        def calc_expected_momentum(months):
            start_date = end_date - pd.DateOffset(months=months)
            start_idx = self.fund_data[self.fund_data['date'] >= start_date].index[0]
            end_idx = self.fund_data[self.fund_data['date'] <= end_date].index[-1]
            return (self.fund_data['nav'].iloc[end_idx] / self.fund_data['nav'].iloc[start_idx] - 1) * 100
        
        expected_3m = calc_expected_momentum(3)
        expected_6m = calc_expected_momentum(6)
        expected_12m = calc_expected_momentum(12)
        
        # Check that the calculated values are close to the expected values
        self.assertAlmostEqual(momentum_3m, expected_3m, places=4)
        self.assertAlmostEqual(momentum_6m, expected_6m, places=4)
        self.assertAlmostEqual(momentum_12m, expected_12m, places=4)

if __name__ == '__main__':
    unittest.main()