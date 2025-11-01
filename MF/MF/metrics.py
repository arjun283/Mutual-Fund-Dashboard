import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

def calculate_all_metrics(df_funds, df_benchmark, risk_free_rate, analysis_period_years=3):
    """
    Calculate all financial metrics for each fund in the dataset.
    
    Parameters:
    -----------
    df_funds : pandas.DataFrame
        DataFrame containing fund data with columns: date, ticker, nav, fund_name
    df_benchmark : pandas.DataFrame
        DataFrame containing benchmark data with columns: date, nav
    risk_free_rate : float
        Annual risk-free rate (decimal form, e.g., 0.04 for 4%)
    analysis_period_years : int, optional (default=3)
        Number of years to analyze
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all calculated metrics for each fund
    """
    # Get unique fund tickers
    tickers = df_funds['ticker'].unique()
    
    # Initialize results DataFrame
    results = []
    
    # Calculate metrics for each fund
    for ticker in tickers:
        # Get fund data
        fund_data = df_funds[df_funds['ticker'] == ticker].sort_values('date')
        
        # Get fund name
        fund_name = fund_data['fund_name'].iloc[0] if 'fund_name' in fund_data.columns else ticker
        
        # Calculate returns
        returns = calculate_returns(fund_data)
        
        # Calculate benchmark returns for the same period
        benchmark_returns = calculate_benchmark_returns(df_benchmark, fund_data)
        
        # Calculate metrics
        annualized_return = calculate_annualized_return(returns)
        volatility = calculate_volatility(returns)
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = calculate_sortino_ratio(returns, risk_free_rate)
        max_drawdown = calculate_max_drawdown(fund_data)
        beta, alpha = calculate_capm_metrics(returns, benchmark_returns, risk_free_rate)
        momentum_3m, momentum_6m, momentum_12m = calculate_momentum(fund_data)
        liquidity = calculate_liquidity_proxy(fund_data)
        market_sentiment = calculate_market_sentiment(df_benchmark)
        
        # Compile results
        result = {
            'Ticker': ticker,
            'Fund Name': fund_name,
            'Annualized Return (%)': annualized_return * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown (%)': max_drawdown * 100,
            'Beta': beta,
            'Alpha': alpha * 100,  # Convert to percentage
            '3M Momentum': momentum_3m * 100,
            '6M Momentum': momentum_6m * 100,
            '12M Momentum': momentum_12m * 100,
            'Liquidity Score': liquidity,
            'Market Sentiment': market_sentiment
        }
        
        results.append(result)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    return df_results

def calculate_returns(fund_data, period='daily'):
    """
    Calculate returns for a fund.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
    period : str, optional (default='daily')
        Return calculation period ('daily', 'weekly', 'monthly')
        
    Returns:
    --------
    pandas.Series
        Series of returns
    """
    # Sort data by date
    fund_data = fund_data.sort_values('date')
    
    # Calculate returns based on period
    if period == 'daily':
        returns = fund_data['nav'].pct_change().dropna()
    elif period == 'weekly':
        fund_data = fund_data.set_index('date')
        weekly_data = fund_data['nav'].resample('W').last()
        returns = weekly_data.pct_change().dropna()
    elif period == 'monthly':
        fund_data = fund_data.set_index('date')
        monthly_data = fund_data['nav'].resample('M').last()
        returns = monthly_data.pct_change().dropna()
    else:
        raise ValueError(f"Invalid period: {period}. Use 'daily', 'weekly', or 'monthly'.")
    
    return returns

def calculate_benchmark_returns(df_benchmark, fund_data, period='daily'):
    """
    Calculate benchmark returns for the same period as fund data.
    
    Parameters:
    -----------
    df_benchmark : pandas.DataFrame
        DataFrame containing benchmark data with columns: date, nav
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
    period : str, optional (default='daily')
        Return calculation period ('daily', 'weekly', 'monthly')
        
    Returns:
    --------
    pandas.Series
        Series of benchmark returns
    """
    # Get min and max dates from fund data
    min_date = fund_data['date'].min()
    max_date = fund_data['date'].max()
    
    # Filter benchmark data for the same period
    benchmark_data = df_benchmark[
        (df_benchmark['date'] >= min_date) & 
        (df_benchmark['date'] <= max_date)
    ].copy()
    
    # Calculate benchmark returns
    return calculate_returns(benchmark_data, period)

def calculate_annualized_return(returns, periods_per_year=252):
    """
    Calculate annualized return from a series of returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    periods_per_year : int, optional (default=252)
        Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
    --------
    float
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative return
    cumulative_return = (1 + returns).prod() - 1
    
    # Calculate number of periods
    n_periods = len(returns)
    
    # Calculate years
    years = n_periods / periods_per_year
    
    # Calculate annualized return
    annualized_return = (1 + cumulative_return) ** (1 / years) - 1
    
    return annualized_return

def calculate_volatility(returns, periods_per_year=252):
    """
    Calculate annualized volatility from a series of returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    periods_per_year : int, optional (default=252)
        Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
    --------
    float
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate standard deviation of returns
    std_dev = returns.std()
    
    # Annualize volatility
    annualized_vol = std_dev * np.sqrt(periods_per_year)
    
    return annualized_vol

def calculate_rolling_volatility(fund_data, window=30):
    """
    Calculate rolling volatility for a fund.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
    window : int, optional (default=30)
        Rolling window size in days
        
    Returns:
    --------
    pandas.Series
        Series of rolling volatility values
    """
    # Sort data by date
    fund_data = fund_data.sort_values('date').set_index('date')
    
    # Calculate daily returns
    returns = fund_data['nav'].pct_change().dropna()
    
    # Calculate rolling standard deviation
    rolling_std = returns.rolling(window=window).std()
    
    # Annualize volatility
    rolling_vol = rolling_std * np.sqrt(252)
    
    return rolling_vol

def calculate_sharpe_ratio(a, b=None, risk_free_rate=0.03, periods_per_year=252):
    """
    Flexible Sharpe ratio calculation that supports two calling patterns:

    1) calculate_sharpe_ratio(returns_series, risk_free_rate=0.03)
       - where `a` is a pandas Series of returns and `b` is unused.
    2) calculate_sharpe_ratio(annualized_return, volatility, risk_free_rate=0.03)
       - where `a` is a numeric annualized return and `b` is numeric volatility.

    This keeps backwards compatibility with different callers (tests expect pattern 2).
    """
    # Case 1: `a` is a returns series
    if isinstance(a, (pd.Series, np.ndarray)):
        returns = a
        # If caller passed a numeric `b` as risk-free rate positionally, treat it as risk_free_rate
        rf = b if isinstance(b, (int, float)) else risk_free_rate

        if len(returns) < 2:
            return 0.0

        excess_return = calculate_annualized_return(returns, periods_per_year) - rf
        volatility = calculate_volatility(returns, periods_per_year)
        if volatility == 0:
            return 0.0
        return excess_return / volatility

    # Case 2: caller passed numeric annualized return and volatility
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        ann_return = a
        vol = b
        if vol == 0:
            return 0.0
        return (ann_return - risk_free_rate) / vol

    # Fallback
    raise TypeError('Invalid arguments for calculate_sharpe_ratio')

def calculate_sortino_ratio(returns, ann_return_or_risk_free=None, risk_free_rate=0.03, periods_per_year=252, target_return=0):
    """
    Calculate Sortino ratio from a series of returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    risk_free_rate : float
        Annual risk-free rate (decimal form, e.g., 0.04 for 4%)
    periods_per_year : int, optional (default=252)
        Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)
    target_return : float, optional (default=0)
        Target return threshold for downside deviation
        
    Returns:
    --------
    float
        Sortino ratio
    """
    """
    Flexible Sortino ratio calculation. Supports two calling patterns used in the
    codebase/tests:

    1) calculate_sortino_ratio(returns_series, risk_free_rate=0.03)
       - where only `returns` are provided; annualized return will be computed.
    2) calculate_sortino_ratio(returns_series, annualized_return, risk_free_rate=0.03)
       - where the caller supplies the annualized return explicitly (tests use this).
    """
    # Validate input
    if not isinstance(returns, (pd.Series, np.ndarray)):
        raise TypeError('First argument must be a returns series')

    # Determine annualized return and risk-free
    if isinstance(ann_return_or_risk_free, (int, float)):
        ann_return = ann_return_or_risk_free
        rf = risk_free_rate
    else:
        ann_return = calculate_annualized_return(returns, periods_per_year)
        rf = ann_return_or_risk_free if isinstance(ann_return_or_risk_free, (int, float)) else risk_free_rate

    # Calculate excess return
    excess_return = ann_return - rf

    # Calculate downside deviation using only downside observations
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.inf if excess_return > 0 else 0.0

    # Use mean of squared downside returns (divide by number of downside observations)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(periods_per_year)

    if downside_deviation == 0:
        return 0.0

    return excess_return / downside_deviation

def calculate_max_drawdown(fund_data_or_returns):
    """
    Calculate maximum drawdown for a fund.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
        
    Returns:
    --------
    float
        Maximum drawdown (as a decimal, e.g., 0.25 for 25%)
    """
    # This function accepts either a DataFrame with 'nav' (fund_data)
    # or a pandas Series/array of returns.
    # If passed returns, compute cumulative returns and drawdowns from them.
    if isinstance(fund_data_or_returns, (pd.Series, np.ndarray)):
        returns = pd.Series(fund_data_or_returns)
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns.min()

    # Otherwise assume DataFrame with 'nav'
    fund_data = fund_data_or_returns.sort_values('date')
    nav_series = fund_data['nav'].values
    cum_max = np.maximum.accumulate(nav_series)
    drawdown = (nav_series - cum_max) / cum_max
    return np.min(drawdown)

def calculate_capm_metrics(fund_returns, benchmark_returns, risk_free_rate, periods_per_year=252):
    """
    Calculate CAPM metrics (beta and alpha) for a fund.
    
    Parameters:
    -----------
    fund_returns : pandas.Series
        Series of fund returns
    benchmark_returns : pandas.Series
        Series of benchmark returns
    risk_free_rate : float
        Annual risk-free rate (decimal form, e.g., 0.04 for 4%)
    periods_per_year : int, optional (default=252)
        Number of periods in a year (252 for daily, 52 for weekly, 12 for monthly)
        
    Returns:
    --------
    tuple
        (beta, alpha)
    """
    # Ensure both series have the same index
    if isinstance(fund_returns.index, pd.DatetimeIndex) and isinstance(benchmark_returns.index, pd.DatetimeIndex):
        # Align dates
        common_dates = fund_returns.index.intersection(benchmark_returns.index)
        fund_returns = fund_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
    
    # Check if we have enough data points
    if len(fund_returns) < 10 or len(benchmark_returns) < 10:
        return 1.0, 0.0  # Default values if not enough data
    
    # Calculate daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    fund_excess_returns = fund_returns - daily_rf
    benchmark_excess_returns = benchmark_returns - daily_rf
    
    # Prepare data for regression
    X = sm.add_constant(benchmark_excess_returns)
    
    try:
        # Run regression
        model = sm.OLS(fund_excess_returns, X).fit()
        
        # Extract beta and alpha
        beta = model.params[1]
        alpha = model.params[0] * periods_per_year  # Annualize alpha
        
        return beta, alpha
    except Exception as e:
        print(f"Error in CAPM calculation: {str(e)}")
        return 1.0, 0.0  # Default values if regression fails


def calculate_beta_alpha(fund_returns, benchmark_returns, start_date=None, end_date=None):
    """
    Compatibility helper used by the tests. Performs a simple linear regression
    of fund_returns on benchmark_returns and returns (beta, alpha_annualized).

    Parameters match what the unit tests expect.
    """
    # Convert to numpy arrays and align if pandas Series with datetime indices
    if isinstance(fund_returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
        # Align by index
        common = fund_returns.index.intersection(benchmark_returns.index)
        y = fund_returns.loc[common].values
        X = benchmark_returns.loc[common].values.reshape(-1, 1)
    else:
        y = np.asarray(fund_returns)
        X = np.asarray(benchmark_returns).reshape(-1, 1)

    # Add constant for intercept
    X_with_const = np.column_stack([np.ones(X.shape[0]), X])
    coeffs, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
    intercept = coeffs[0]
    slope = coeffs[1] if len(coeffs) > 1 else 0.0

    alpha_annualized = intercept * 252  # annualize intercept like tests expect
    beta = slope

    return beta, alpha_annualized

def get_capm_regression_data(fund_data, benchmark_data):
    """
    Prepare data for CAPM regression visualization.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
    benchmark_data : pandas.DataFrame
        DataFrame containing benchmark data with columns: date, nav
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with fund and benchmark returns
    """
    # Calculate returns
    fund_returns = calculate_returns(fund_data)
    
    # Ensure benchmark data covers the same period
    min_date = fund_data['date'].min()
    max_date = fund_data['date'].max()
    
    benchmark_filtered = benchmark_data[
        (benchmark_data['date'] >= min_date) & 
        (benchmark_data['date'] <= max_date)
    ].copy()
    
    benchmark_returns = calculate_returns(benchmark_filtered)
    
    # Align dates if using DatetimeIndex
    if isinstance(fund_returns.index, pd.DatetimeIndex) and isinstance(benchmark_returns.index, pd.DatetimeIndex):
        common_dates = fund_returns.index.intersection(benchmark_returns.index)
        fund_returns = fund_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
    
    # Create DataFrame with both returns
    df_regression = pd.DataFrame({
        'fund_return': fund_returns,
        'benchmark_return': benchmark_returns
    })
    
    return df_regression

def calculate_momentum(fund_data, end_date=None, months=None, periods=[3, 6, 12]):
    """
    Calculate momentum metrics for a fund.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
    periods : list, optional (default=[3, 6, 12])
        List of periods (in months) to calculate momentum for
        
    Returns:
    --------
    tuple
        (3-month momentum, 6-month momentum, 12-month momentum)
    """
    # Sort data by date
    fund_data = fund_data.sort_values('date').reset_index(drop=True)

    # If caller provided an end_date and months, compute single-month momentum (used by tests)
    if end_date is not None and months is not None:
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - pd.DateOffset(months=months)

        # Find first index >= start_dt and last index <= end_dt
        start_candidates = fund_data[fund_data['date'] >= start_dt]
        end_candidates = fund_data[fund_data['date'] <= end_dt]

        if start_candidates.empty or end_candidates.empty:
            return 0.0

        start_idx = start_candidates.index[0]
        end_idx = end_candidates.index[-1]

        start_nav = fund_data.loc[start_idx, 'nav']
        end_nav = fund_data.loc[end_idx, 'nav']

        return (end_nav / start_nav - 1) * 100

    # Otherwise compute tuple of momentum values for supplied periods (in months)
    latest_nav = fund_data['nav'].iloc[-1]
    momentum_values = []
    for period in periods:
        start_dt = fund_data['date'].iloc[-1] - pd.DateOffset(months=period)
        start_candidates = fund_data[fund_data['date'] >= start_dt]
        if start_candidates.empty:
            momentum_values.append(0.0)
            continue
        start_nav = start_candidates.iloc[0]['nav']
        momentum_values.append((latest_nav / start_nav) - 1)

    return tuple(momentum_values)

def calculate_liquidity_proxy(fund_data):
    """
    Calculate a liquidity proxy for a fund based on NAV update frequency.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
        
    Returns:
    --------
    float
        Liquidity score (0-1)
    """
    # Sort data by date
    fund_data = fund_data.sort_values('date')
    
    # Check if we have enough data
    if len(fund_data) < 10:  # Require at least 10 data points
        return 0.5  # Default value if not enough data
    
    # Calculate average gap between updates (in days)
    fund_data['date_diff'] = fund_data['date'].diff().dt.days
    avg_gap = fund_data['date_diff'].mean()
    
    # Calculate NAV change frequency
    fund_data['nav_change'] = fund_data['nav'].pct_change().abs()
    nav_change_freq = (fund_data['nav_change'] > 0.0001).mean()
    
    # Combine metrics into a liquidity score (higher is better)
    # Lower gap and higher change frequency indicate better liquidity
    if np.isnan(avg_gap) or avg_gap == 0:
        gap_score = 1.0
    else:
        gap_score = 1.0 / (1.0 + avg_gap)  # Normalize to 0-1 range
    
    # Combine scores (equal weight)
    liquidity_score = 0.5 * gap_score + 0.5 * nav_change_freq
    
    return liquidity_score

def calculate_market_sentiment(benchmark_data):
    """
    Calculate a simple market sentiment indicator based on recent benchmark performance.
    
    Parameters:
    -----------
    benchmark_data : pandas.DataFrame
        DataFrame containing benchmark data with columns: date, nav
        
    Returns:
    --------
    float
        Sentiment score (-1 to 1, where positive is bullish)
    """
    # Sort data by date
    benchmark_data = benchmark_data.sort_values('date')
    
    # Check if we have enough data
    if len(benchmark_data) < 90:  # Need at least 90 days for 3-month analysis
        return 0.0  # Neutral sentiment if not enough data
    
    # Get last 3 months of data (approximately 63 trading days)
    recent_data = benchmark_data.tail(63)
    
    # Calculate 3-month return
    start_nav = recent_data['nav'].iloc[0]
    end_nav = recent_data['nav'].iloc[-1]
    return_3m = (end_nav / start_nav) - 1
    
    # Calculate recent volatility (last month vs previous 2 months)
    recent_month = benchmark_data.tail(21)
    previous_months = benchmark_data.iloc[-63:-21]
    
    recent_vol = recent_month['nav'].pct_change().std()
    previous_vol = previous_months['nav'].pct_change().std()
    
    # Volatility spike indicator (-1 to 1)
    if previous_vol == 0:
        vol_indicator = 0.0
    else:
        vol_ratio = recent_vol / previous_vol
        vol_indicator = -np.tanh(vol_ratio - 1)  # Negative for volatility spike
    
    # Combine return sign and volatility indicator
    return_sign = 1 if return_3m > 0 else -1
    sentiment = 0.7 * return_sign + 0.3 * vol_indicator
    
    return sentiment

def calculate_benchmark_metrics(df_benchmark, risk_free_rate):
    """
    Calculate key metrics for the benchmark index.
    
    Parameters:
    -----------
    df_benchmark : pandas.DataFrame
        DataFrame containing benchmark data with columns: date, nav
    risk_free_rate : float
        Annual risk-free rate (decimal form, e.g., 0.04 for 4%)
        
    Returns:
    --------
    dict
        Dictionary with benchmark metrics
    """
    # Calculate benchmark returns
    benchmark_returns = calculate_returns(df_benchmark)
    
    # Calculate metrics
    annualized_return = calculate_annualized_return(benchmark_returns)
    volatility = calculate_volatility(benchmark_returns)
    sharpe = calculate_sharpe_ratio(benchmark_returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(df_benchmark)
    
    # Compile results
    metrics = {
        'Annualized Return (%)': annualized_return * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown * 100
    }
    
    return metrics

def calculate_metrics(fund_data, benchmark_data, risk_free_rate=0.04):
    """
    Calculate key metrics for a fund compared to a benchmark.
    
    Parameters:
    -----------
    fund_data : pandas.DataFrame
        DataFrame containing fund data with columns: date, nav
    benchmark_data : pandas.DataFrame
        DataFrame containing benchmark data with columns: date, nav
    risk_free_rate : float, optional (default=0.04)
        Annual risk-free rate (decimal form, e.g., 0.04 for 4%)
        
    Returns:
    --------
    dict
        Dictionary with calculated metrics
    """
    # Calculate returns
    fund_returns = calculate_returns(fund_data)
    benchmark_returns = calculate_benchmark_returns(benchmark_data, fund_data)
    
    # Calculate metrics
    annualized_return = calculate_annualized_return(fund_returns)
    volatility = calculate_volatility(fund_returns)
    sharpe = calculate_sharpe_ratio(fund_returns, risk_free_rate)
    sortino = calculate_sortino_ratio(fund_returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(fund_data)
    beta, alpha = calculate_capm_metrics(fund_returns, benchmark_returns, risk_free_rate)
    momentum_3m, momentum_6m, momentum_12m = calculate_momentum(fund_data)
    liquidity = calculate_liquidity_proxy(fund_data)
    market_sentiment = calculate_market_sentiment(benchmark_data)
    
    # Compile results
    metrics = {
        'Annualized Return (%)': annualized_return * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown (%)': max_drawdown * 100,
        'Beta': beta,
        'Alpha': alpha * 100,  # Convert to percentage
        '3M Momentum': momentum_3m * 100,
        '6M Momentum': momentum_6m * 100,
        '12M Momentum': momentum_12m * 100,
        'Liquidity Score': liquidity,
        'Market Sentiment': market_sentiment
    }
    
    return metrics