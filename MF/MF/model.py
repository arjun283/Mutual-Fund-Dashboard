import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Define risk tolerance weights for different metrics
RISK_WEIGHTS = {
    'low': {
        'Annualized Return (%)': 0.10,
        'Volatility (%)': -0.20,
        'Sharpe Ratio': 0.15,
        'Sortino Ratio': 0.20,
        'Max Drawdown (%)': -0.20,
        'Beta': -0.15,
        '3M Momentum': 0.00,
        '6M Momentum': 0.05,
        '12M Momentum': 0.05,
        'Liquidity Score': 0.05,
        'Market Sentiment': -0.05  # Negative weight means inverse relationship
    },
    'medium': {
        'Annualized Return (%)': 0.20,
        'Volatility (%)': -0.10,
        'Sharpe Ratio': 0.20,
        'Sortino Ratio': 0.15,
        'Max Drawdown (%)': -0.10,
        'Beta': -0.05,
        '3M Momentum': 0.05,
        '6M Momentum': 0.10,
        '12M Momentum': 0.05,
        'Liquidity Score': 0.05,
        'Market Sentiment': 0.05
    },
    'high': {
        'Annualized Return (%)': 0.25,
        'Volatility (%)': -0.05,
        'Sharpe Ratio': 0.10,
        'Sortino Ratio': 0.05,
        'Max Drawdown (%)': -0.05,
        'Beta': 0.05,
        '3M Momentum': 0.15,
        '6M Momentum': 0.15,
        '12M Momentum': 0.10,
        'Liquidity Score': 0.05,
        'Market Sentiment': 0.10
    }
}

def rank_funds(funds_metrics, risk_tolerance, top_n=5, use_ml_model=False):
    """
    Rank funds based on a composite score calculated from various metrics.
    
    Parameters:
    -----------
    funds_metrics : pandas.DataFrame
        DataFrame containing fund metrics
    risk_tolerance : str
        Risk tolerance level ('low', 'medium', or 'high')
    top_n : int, optional (default=5)
        Number of top funds to return
    use_ml_model : bool, optional (default=False)
        Whether to use ML model for ranking
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with top N ranked funds
    """
    # Check if we have enough funds to rank
    if len(funds_metrics) == 0:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = funds_metrics.copy()
    
    # Get weights based on risk tolerance
    if risk_tolerance not in RISK_WEIGHTS:
        raise ValueError(f"Invalid risk tolerance: {risk_tolerance}. Use 'low', 'medium', or 'high'.")
    
    weights = RISK_WEIGHTS[risk_tolerance]
    
    # Normalize metrics for scoring
    metrics_to_normalize = list(weights.keys())
    available_metrics = [m for m in metrics_to_normalize if m in df.columns]
    
    # Check if we have enough metrics
    if len(available_metrics) < 3:
        raise ValueError("Not enough metrics available for ranking.")
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Normalize each metric
    for metric in available_metrics:
        # Skip metrics with all zeros or NaNs
        if df[metric].isna().all() or (df[metric] == 0).all():
            continue
        
        # Fill NaNs with median
        df[metric] = df[metric].fillna(df[metric].median())
        
        # Normalize
        df[f"{metric}_normalized"] = scaler.fit_transform(df[[metric]])
        
        # Invert normalization for metrics where lower is better
        if weights[metric] < 0:
            df[f"{metric}_normalized"] = 1 - df[f"{metric}_normalized"]
    
    # Calculate composite score
    df['Composite Score'] = 0
    
    for metric in available_metrics:
        weight = abs(weights[metric])  # Use absolute weight value
        if f"{metric}_normalized" in df.columns:
            df['Composite Score'] += weight * df[f"{metric}_normalized"]
    
    # Normalize composite score
    max_score = df['Composite Score'].max()
    min_score = df['Composite Score'].min()
    
    if max_score > min_score:
        df['Composite Score'] = (df['Composite Score'] - min_score) / (max_score - min_score)
    
    # Use ML model if requested
    if use_ml_model and len(df) >= 10:  # Need enough data for ML
        try:
            ml_scores = predict_future_returns(df)
            # Combine with composite score (equal weight)
            df['ML Score'] = ml_scores
            df['Composite Score'] = 0.7 * df['Composite Score'] + 0.3 * df['ML Score']
        except Exception as e:
            print(f"Error using ML model: {str(e)}. Using only composite score.")
    
    # Sort by composite score
    df = df.sort_values('Composite Score', ascending=False)
    
    # Return top N funds
    return df.head(top_n)

def predict_future_returns(funds_metrics):
    """
    Use a machine learning model to predict future returns based on current metrics.
    This is a simplified model for educational purposes.
    
    Parameters:
    -----------
    funds_metrics : pandas.DataFrame
        DataFrame containing fund metrics
        
    Returns:
    --------
    numpy.ndarray
        Normalized predicted returns
    """
    # Select features and target
    features = [
        'Annualized Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio',
        'Max Drawdown (%)', 'Beta', 'Alpha', '3M Momentum', '6M Momentum', '12M Momentum',
        'Liquidity Score'
    ]
    
    # Filter available features
    available_features = [f for f in features if f in funds_metrics.columns]
    
    # Check if we have enough features
    if len(available_features) < 3:
        raise ValueError("Not enough features available for ML model.")
    
    # Prepare data
    X = funds_metrics[available_features].fillna(0)
    
    # Since we don't have actual future returns, we'll use a simple model
    # that predicts based on a combination of existing metrics
    model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    
    # Create a synthetic target based on Sharpe ratio and momentum
    # This is just for demonstration - in a real scenario, you would use actual future returns
    y_synthetic = (
        0.4 * funds_metrics['Sharpe Ratio'].fillna(0) + 
        0.3 * funds_metrics['3M Momentum'].fillna(0) + 
        0.2 * funds_metrics['6M Momentum'].fillna(0) + 
        0.1 * funds_metrics['12M Momentum'].fillna(0)
    )
    
    # Train model
    model.fit(X, y_synthetic)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Normalize predictions to 0-1 range
    min_pred = predictions.min()
    max_pred = predictions.max()
    
    if max_pred > min_pred:
        normalized_predictions = (predictions - min_pred) / (max_pred - min_pred)
    else:
        normalized_predictions = np.ones(len(predictions)) * 0.5
    
    return normalized_predictions

def train_ml_model(historical_data, future_returns):
    """
    Train a machine learning model to predict future returns.
    
    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing historical fund metrics
    future_returns : pandas.Series
        Series containing actual future returns
        
    Returns:
    --------
    tuple
        (trained model, feature importance, cross-validation scores)
    """
    # Select features
    features = [
        'Annualized Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Sortino Ratio',
        'Max Drawdown (%)', 'Beta', 'Alpha', '3M Momentum', '6M Momentum', '12M Momentum',
        'Liquidity Score'
    ]
    
    # Filter available features
    available_features = [f for f in features if f in historical_data.columns]
    
    # Prepare data
    X = historical_data[available_features].fillna(0)
    y = future_returns
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    best_model_name = None
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'cv_rmse': cv_rmse
        }
        
        # Check if this is the best model
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name
    
    # Get feature importance for the best model
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
    else:
        feature_importance = pd.DataFrame()
    
    return best_model, feature_importance, results

def suggest_allocation(ranked_funds, invest_amount, risk_tolerance):
    """
    Suggest allocation of investment amount across ranked funds.
    
    Parameters:
    -----------
    ranked_funds : pandas.DataFrame
        DataFrame containing ranked funds
    invest_amount : float
        Total investment amount
    risk_tolerance : str
        Risk tolerance level ('low', 'medium', or 'high')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with suggested allocation
    """
    # Check if we have funds to allocate
    if len(ranked_funds) == 0:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    df = ranked_funds.copy()
    
    # Calculate weights based on composite score
    total_score = df['Composite Score'].sum()
    
    if total_score > 0:
        df['Weight'] = df['Composite Score'] / total_score
    else:
        # Equal weights if all scores are zero
        df['Weight'] = 1.0 / len(df)
    
    # Adjust weights based on risk tolerance
    if risk_tolerance == 'low':
        # For low risk, make allocation more conservative by reducing weight of high-volatility funds
        volatility_factor = 1.0 - df['Volatility (%)'] / 100.0
        volatility_factor = volatility_factor.clip(0.5, 1.0)  # Limit the adjustment
        df['Weight'] = df['Weight'] * volatility_factor
        # Renormalize
        df['Weight'] = df['Weight'] / df['Weight'].sum()
    
    elif risk_tolerance == 'high':
        # For high risk, increase weight of high-momentum funds
        momentum_factor = 1.0 + df['12M Momentum'] / 100.0
        momentum_factor = momentum_factor.clip(1.0, 1.5)  # Limit the adjustment
        df['Weight'] = df['Weight'] * momentum_factor
        # Renormalize
        df['Weight'] = df['Weight'] / df['Weight'].sum()
    
    # Calculate amount for each fund
    df['Amount'] = df['Weight'] * invest_amount
    
    # Create a display weight column as formatted string
    df['Weight Display'] = df['Weight'].apply(lambda x: f"{x:.2%}")
    
    # Select columns for output
    allocation = df[['Fund Name', 'Amount', 'Weight', 'Weight Display']].copy()
    
    return allocation

def simulate_portfolio_outcomes(ranked_funds, weights, invest_amount, time_horizon_years, num_simulations=1000):
    """
    Simulate portfolio outcomes using Monte Carlo simulation.
    
    Parameters:
    -----------
    ranked_funds : pandas.DataFrame
        DataFrame containing ranked funds
    weights : numpy.ndarray
        Array of portfolio weights
    invest_amount : float
        Initial investment amount
    time_horizon_years : int
        Investment time horizon in years
    num_simulations : int, optional (default=1000)
        Number of Monte Carlo simulations
        
    Returns:
    --------
    dict
        Dictionary with simulation results
    """
    # Check if we have funds to simulate
    if len(ranked_funds) == 0 or len(weights) == 0:
        return {
            'time': np.arange(time_horizon_years + 1),
            'median': np.ones(time_horizon_years + 1) * invest_amount,
            'lower': np.ones(time_horizon_years + 1) * invest_amount,
            'upper': np.ones(time_horizon_years + 1) * invest_amount
        }
    
    # Extract expected returns and volatilities
    returns = ranked_funds['Annualized Return (%)'].values / 100.0
    volatilities = ranked_funds['Volatility (%)'].values / 100.0
    
    # Ensure weights sum to 1
    weights = np.array(weights) / np.sum(weights)
    
    # Calculate portfolio expected return and volatility
    portfolio_return = np.sum(returns * weights)
    
    # Simplified covariance matrix (assuming correlation of 0.5 between all funds)
    correlation = 0.5
    covariance_matrix = np.outer(volatilities, volatilities) * correlation
    np.fill_diagonal(covariance_matrix, volatilities ** 2)
    
    portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_variance)
    
    # Simulate portfolio outcomes
    time_points = np.arange(time_horizon_years + 1)
    simulations = np.zeros((num_simulations, len(time_points)))
    simulations[:, 0] = invest_amount  # Initial investment
    
    for i in range(num_simulations):
        # Generate random annual returns
        annual_returns = np.random.normal(
            portfolio_return, 
            portfolio_volatility, 
            time_horizon_years
        )
        
        # Calculate cumulative portfolio value
        portfolio_value = invest_amount
        for j in range(time_horizon_years):
            portfolio_value *= (1 + annual_returns[j])
            simulations[i, j+1] = portfolio_value
    
    # Calculate statistics
    median_values = np.median(simulations, axis=0)
    lower_percentile = np.percentile(simulations, 10, axis=0)  # 10th percentile
    upper_percentile = np.percentile(simulations, 90, axis=0)  # 90th percentile
    
    return {
        'time': time_points,
        'median': median_values,
        'lower': lower_percentile,
        'upper': upper_percentile
    }

def efficient_frontier_approximation(volatilities, returns, x_range):
    """
    Generate an approximation of the efficient frontier.
    
    Parameters:
    -----------
    volatilities : numpy.ndarray
        Array of fund volatilities
    returns : numpy.ndarray
        Array of fund returns
    x_range : numpy.ndarray
        Array of x-values for the frontier
        
    Returns:
    --------
    numpy.ndarray
        Array of y-values for the frontier
    """
    # Fit a quadratic function to the data points
    coeffs = np.polyfit(volatilities, returns, 2)
    
    # Generate y-values using the fitted function
    y_range = np.polyval(coeffs, x_range)
    
    # Ensure the frontier is concave (efficient frontier shape)
    if coeffs[0] > 0:  # If the quadratic term is positive, the curve is convex
        # Invert the curve to make it concave
        max_y = y_range.max()
        y_range = max_y - (y_range - y_range.min())
    
    return y_range