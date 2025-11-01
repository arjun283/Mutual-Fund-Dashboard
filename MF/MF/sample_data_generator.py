import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data(num_funds=10, start_date='2020-01-01', end_date='2022-12-31', seed=42):
    """
    Generate sample mutual fund data for testing.
    
    Parameters:
    -----------
    num_funds : int, optional (default=10)
        Number of funds to generate
    start_date : str, optional (default='2020-01-01')
        Start date for the data
    end_date : str, optional (default='2022-12-31')
        End date for the data
    seed : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary of DataFrames containing fund data, with fund tickers as keys
    """
    np.random.seed(seed)
    
    # Convert dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Generate dates (business days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Fund characteristics
    fund_types = ['Large Cap', 'Mid Cap', 'Small Cap', 'Balanced', 'Growth', 'Value', 'Dividend', 'Sector', 'International', 'Bond']
    fund_tickers = [f'FUND{i+1}' for i in range(num_funds)]
    fund_names = [f'{np.random.choice(fund_types)} Fund {i+1}' for i in range(num_funds)]
    
    # Generate different characteristics for each fund
    mean_returns = np.random.uniform(0.0003, 0.0008, num_funds)  # Daily mean returns between 7.5% and 20% annualized
    volatilities = np.random.uniform(0.005, 0.015, num_funds)    # Daily volatility between 8% and 24% annualized
    betas = np.random.uniform(0.7, 1.3, num_funds)               # Beta relative to benchmark
    
    # Generate benchmark data
    benchmark_mean_return = 0.0005  # ~12.5% annualized
    benchmark_volatility = 0.008    # ~12.7% annualized
    
    benchmark_returns = np.random.normal(benchmark_mean_return, benchmark_volatility, n_days)
    benchmark_prices = 100 * (1 + benchmark_returns).cumprod()
    
    benchmark_data = pd.DataFrame({
        'date': dates,
        'nav': benchmark_prices
    })
    
    # Generate fund data
    funds_data = {}
    
    for i in range(num_funds):
        # Generate correlated returns with the benchmark
        z = np.random.normal(0, 1, n_days)
        fund_specific_returns = betas[i] * benchmark_returns + volatilities[i] * z
        fund_specific_returns = fund_specific_returns - fund_specific_returns.mean() + mean_returns[i]  # Adjust mean
        
        # Add some autocorrelation for momentum effects
        for j in range(1, n_days):
            fund_specific_returns[j] = 0.05 * fund_specific_returns[j-1] + 0.95 * fund_specific_returns[j]
        
        # Convert returns to prices
        fund_prices = 100 * (1 + fund_specific_returns).cumprod()
        
        # Add some missing values randomly (about 5%)
        mask = np.random.random(n_days) > 0.05
        fund_prices_with_gaps = fund_prices.copy()
        fund_prices_with_gaps[~mask] = np.nan
        
        # Forward fill missing values
        fund_prices_with_gaps = pd.Series(fund_prices_with_gaps).fillna(method='ffill').values
        
        # Create DataFrame
        fund_data = pd.DataFrame({
            'date': dates,
            'nav': fund_prices_with_gaps
        })
        
        funds_data[fund_tickers[i]] = fund_data
    
    # Create a metadata DataFrame with fund information
    metadata = pd.DataFrame({
        'Ticker': fund_tickers,
        'Fund Name': fund_names,
        'Fund Type': [name.split(' Fund')[0] for name in fund_names],
        'Expense Ratio': np.random.uniform(0.5, 2.0, num_funds),
        'Inception Date': [start_date - timedelta(days=np.random.randint(365, 3650)) for _ in range(num_funds)],
        'AUM (Millions)': np.random.uniform(100, 10000, num_funds)
    })
    
    return funds_data, benchmark_data, metadata

def save_sample_data(output_dir='sample_data', num_funds=10):
    """
    Generate and save sample data to CSV files.
    
    Parameters:
    -----------
    output_dir : str, optional (default='sample_data')
        Directory to save the data
    num_funds : int, optional (default=10)
        Number of funds to generate
        
    Returns:
    --------
    tuple
        (funds_data, benchmark_data, metadata)
    """
    # Generate data
    funds_data, benchmark_data, metadata = generate_sample_data(num_funds=num_funds)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save benchmark data
    benchmark_data.to_csv(os.path.join(output_dir, 'benchmark.csv'), index=False)
    
    # Save fund data
    all_fund_data = []
    for ticker, data in funds_data.items():
        data['ticker'] = ticker
        all_fund_data.append(data)
    
    # Combine all fund data into a single DataFrame
    combined_data = pd.concat(all_fund_data)
    combined_data.to_csv(os.path.join(output_dir, 'funds.csv'), index=False)
    
    # Save metadata
    metadata.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    
    print(f"Sample data saved to {output_dir}/")
    return funds_data, benchmark_data, metadata

def generate_combined_csv(output_dir='sample_data', num_funds=10):
    """
    Generate and save a combined CSV file with all fund data.
    
    Parameters:
    -----------
    output_dir : str, optional (default='sample_data')
        Directory to save the data
    num_funds : int, optional (default=10)
        Number of funds to generate
        
    Returns:
    --------
    str
        Path to the combined CSV file
    """
    # Generate data
    funds_data, benchmark_data, metadata = generate_sample_data(num_funds=num_funds)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all fund data into a single DataFrame
    all_fund_data = []
    for ticker, data in funds_data.items():
        data = data.copy()
        data['ticker'] = ticker
        all_fund_data.append(data)
    
    combined_data = pd.concat(all_fund_data)
    
    # Add benchmark data
    benchmark_data = benchmark_data.copy()
    benchmark_data['ticker'] = 'BENCHMARK'
    combined_data = pd.concat([combined_data, benchmark_data])
    
    # Save combined data
    output_file = os.path.join(output_dir, 'combined_data.csv')
    combined_data.to_csv(output_file, index=False)
    
    print(f"Combined data saved to {output_file}")
    return output_file

if __name__ == '__main__':
    # Generate and save sample data
    save_sample_data()
    
    # Generate combined CSV
    generate_combined_csv()