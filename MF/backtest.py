import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Support both package imports (when running as a package) and direct script imports
try:
    # Preferred when running as part of the MF package
    from .metrics import calculate_metrics, calculate_returns
    from .model import rank_funds
except Exception:
    # Fallback for running as a script (e.g., streamlit run app.py which does `import backtest`)
    from metrics import calculate_metrics, calculate_returns
    from model import rank_funds

def run_backtest(funds_data, benchmark_data, start_date, end_date, rebalance_freq='yearly', risk_tolerance='medium', top_n=5):
    """
    Run a backtest of the fund ranking strategy.
    
    Parameters:
    -----------
    funds_data : dict
        Dictionary of DataFrames containing fund data, with fund tickers as keys
    benchmark_data : pandas.DataFrame
        DataFrame containing benchmark data
    start_date : datetime
        Start date for the backtest
    end_date : datetime
        End date for the backtest
    rebalance_freq : str, optional (default='yearly')
        Rebalancing frequency ('monthly', 'quarterly', 'yearly')
    risk_tolerance : str, optional (default='medium')
        Risk tolerance level ('low', 'medium', or 'high')
    top_n : int, optional (default=5)
        Number of top funds to include in the portfolio
        
    Returns:
    --------
    dict
        Dictionary containing backtest results
    """
    # Validate inputs
    if rebalance_freq not in ['monthly', 'quarterly', 'yearly']:
        raise ValueError("rebalance_freq must be 'monthly', 'quarterly', or 'yearly'")
    
    if risk_tolerance not in ['low', 'medium', 'high']:
        raise ValueError("risk_tolerance must be 'low', 'medium', or 'high'")
    
    # Set rebalancing parameters
    if rebalance_freq == 'monthly':
        months_delta = 1
    elif rebalance_freq == 'quarterly':
        months_delta = 3
    else:  # yearly
        months_delta = 12
    
    # If a single combined DataFrame of all funds was passed, convert to dict by ticker
    if isinstance(funds_data, pd.DataFrame):
        funds_dict = {}
        for ticker, df in funds_data.groupby('ticker'):
            funds_dict[ticker] = df.sort_values('date').copy()
        funds_data = funds_dict

    # Initialize results
    portfolio_values = []
    benchmark_values = []
    rebalance_dates = []
    portfolio_holdings = []
    
    # Initialize portfolio with equal cash allocation
    initial_value = 100000  # Starting with 100,000 currency units
    current_value = initial_value
    current_holdings = {}
    current_date = start_date
    
    # Run backtest
    while current_date <= end_date:
        # Determine if rebalancing is needed
        if not rebalance_dates or (current_date.month - rebalance_dates[-1].month) % months_delta == 0 and current_date.month != rebalance_dates[-1].month:
            # Get historical data up to current date for analysis
            lookback_start = current_date - timedelta(days=365*3)  # 3 years lookback
            
            # Filter data for the lookback period
            filtered_funds_data = {}
            for ticker, data in funds_data.items():
                # Ensure we have a DataFrame
                if not isinstance(data, pd.DataFrame):
                    # Skip non-DataFrame entries
                    continue
                mask = (data['date'] >= lookback_start) & (data['date'] <= current_date)
                if mask.sum() >= 252:  # Require at least 1 year of data (approx. 252 trading days)
                    filtered_funds_data[ticker] = data.loc[mask].copy()

            # If not enough funds meet the strict lookback, try a relaxed rule for demo/sample data
            if len(filtered_funds_data) < top_n:
                relaxed_filtered = {}
                for ticker, data in funds_data.items():
                    if not isinstance(data, pd.DataFrame):
                        continue
                    mask = (data['date'] >= lookback_start) & (data['date'] <= current_date)
                    # relaxed requirement: at least ~30 trading days
                    if mask.sum() >= 30:
                        relaxed_filtered[ticker] = data.loc[mask].copy()

                if len(relaxed_filtered) >= top_n:
                    filtered_funds_data = relaxed_filtered
                else:
                    # final fallback: accept any fund with any data in the window
                    any_filtered = {}
                    for ticker, data in funds_data.items():
                        if not isinstance(data, pd.DataFrame):
                            continue
                        mask = (data['date'] >= lookback_start) & (data['date'] <= current_date)
                        if mask.sum() > 0:
                            any_filtered[ticker] = data.loc[mask].copy()

                    if len(any_filtered) >= top_n:
                        filtered_funds_data = any_filtered
                    elif len(any_filtered) > 0:
                        # proceed with whatever limited data we have (demo fallback)
                        filtered_funds_data = any_filtered
                    else:
                        # No data in lookback at all; skip forward
                        current_date += timedelta(days=30)
                        continue
            
            # Calculate metrics for each fund
            metrics_list = []
            # Also prepare a benchmark slice for the lookback period
            benchmark_slice = None
            if isinstance(benchmark_data, pd.DataFrame):
                benchmark_slice = benchmark_data[(benchmark_data['date'] >= lookback_start) & (benchmark_data['date'] <= current_date)].copy()

            for ticker, df in filtered_funds_data.items():
                try:
                    # Calculate returns
                    returns = calculate_returns(df)

                    # Use fund DataFrame and benchmark slice to calculate metrics
                    if benchmark_slice is None or benchmark_slice.empty:
                        metrics = calculate_metrics(df, benchmark_data, risk_free_rate=0.03)
                    else:
                        metrics = calculate_metrics(df, benchmark_slice, risk_free_rate=0.03)

                    # Ensure metrics is a dict-like mapping
                    if isinstance(metrics, dict):
                        metrics['Ticker'] = ticker
                        metrics_list.append(metrics)
                    else:
                        print(f"Metrics for {ticker} not in expected format: {type(metrics)}")
                except Exception as e:
                    print(f"Error calculating metrics for {ticker}: {e}")
                    continue
            
            # Create DataFrame from metrics
            if not metrics_list:
                current_date += timedelta(days=30)
                continue
                
            metrics_df = pd.DataFrame(metrics_list)
            
            # Rank funds
            ranked_funds = rank_funds(metrics_df, risk_tolerance)
            
            # Select top N funds
            top_funds = ranked_funds.head(top_n)
            
            # Rebalance portfolio
            # Sell existing holdings; if there are no holdings yet, keep current_value
            current_value = sum(current_holdings.values()) if current_holdings else current_value
            
            # Buy new holdings with equal weight
            new_holdings = {}
            weight_per_fund = 1.0 / len(top_funds)
            for _, fund in top_funds.iterrows():
                ticker = fund['Ticker']
                allocation = current_value * weight_per_fund
                new_holdings[ticker] = allocation
            
            current_holdings = new_holdings
            rebalance_dates.append(current_date)
            portfolio_holdings.append(current_holdings.copy())
        
        # Calculate portfolio value at current date
        portfolio_value = 0
        for ticker, allocation in current_holdings.items():
            # Find the closest date in the fund data
            fund_data = funds_data[ticker]
            closest_idx = (fund_data['date'] - current_date).abs().idxmin()
            closest_date = fund_data.loc[closest_idx, 'date']
            
            # Only use the data if it's within 5 days of the current date
            if abs((closest_date - current_date).days) <= 5:
                nav = fund_data.loc[closest_idx, 'nav']
                shares = allocation / nav
                current_value = shares * nav
                portfolio_value += current_value
        
        # If no valid data found, use the previous portfolio value
        if portfolio_value == 0 and portfolio_values:
            portfolio_value = portfolio_values[-1]
        elif portfolio_value == 0:
            portfolio_value = initial_value
        
        # Calculate benchmark value
        benchmark_idx = (benchmark_data['date'] - current_date).abs().idxmin()
        benchmark_closest_date = benchmark_data.loc[benchmark_idx, 'date']
        
        if abs((benchmark_closest_date - current_date).days) <= 5:
            benchmark_nav = benchmark_data.loc[benchmark_idx, 'nav']
            if not benchmark_values:
                benchmark_shares = initial_value / benchmark_nav
                benchmark_value = benchmark_shares * benchmark_nav
            else:
                benchmark_value = (benchmark_nav / benchmark_data.loc[(benchmark_data['date'] - rebalance_dates[0]).abs().idxmin(), 'nav']) * initial_value
        else:
            benchmark_value = benchmark_values[-1] if benchmark_values else initial_value
        
        # Record values
        portfolio_values.append(portfolio_value)
        benchmark_values.append(benchmark_value)
        
        # Move to next month
        current_date += timedelta(days=30)  # Approximate month
    
    # Create results DataFrame
    dates = [start_date + timedelta(days=30*i) for i in range(len(portfolio_values))]
    results_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'benchmark_value': benchmark_values
    })
    
    # If no results were recorded, return safe empty metrics
    if results_df.empty:
        performance_metrics = {
            'portfolio_annualized_return': 0.0,
            'benchmark_annualized_return': 0.0,
            'portfolio_volatility': 0.0,
            'benchmark_volatility': 0.0,
            'portfolio_sharpe': 0.0,
            'benchmark_sharpe': 0.0,
            'portfolio_max_drawdown': 0.0,
            'benchmark_max_drawdown': 0.0,
        }

        return {
            'results_df': results_df,
            'performance_metrics': performance_metrics,
            'rebalance_dates': rebalance_dates,
            'portfolio_holdings': portfolio_holdings,
            'dates': [],
            'portfolio_cum_return': [],
            'benchmark_cum_return': [],
            'portfolio_return': performance_metrics['portfolio_annualized_return'],
            'benchmark_return': performance_metrics['benchmark_annualized_return'],
            'portfolio_volatility': performance_metrics['portfolio_volatility'],
            'portfolio_sharpe': performance_metrics['portfolio_sharpe'],
            'benchmark_sharpe': performance_metrics['benchmark_sharpe']
        }

    # Calculate performance metrics
    portfolio_returns = results_df['portfolio_value'].pct_change().dropna()
    benchmark_returns = results_df['benchmark_value'].pct_change().dropna()

    # Annualized return
    years = (end_date - start_date).days / 365.25
    portfolio_annualized_return = ((results_df['portfolio_value'].iloc[-1] / results_df['portfolio_value'].iloc[0]) ** (1/years)) - 1
    benchmark_annualized_return = ((results_df['benchmark_value'].iloc[-1] / results_df['benchmark_value'].iloc[0]) ** (1/years)) - 1
    
    # Annualized volatility
    portfolio_volatility = portfolio_returns.std() * np.sqrt(12)  # Monthly to annual
    benchmark_volatility = benchmark_returns.std() * np.sqrt(12)  # Monthly to annual
    
    # Sharpe ratio (assuming risk-free rate of 0.03)
    risk_free_rate = 0.03
    portfolio_sharpe = (portfolio_annualized_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
    benchmark_sharpe = (benchmark_annualized_return - risk_free_rate) / benchmark_volatility if benchmark_volatility != 0 else 0
    
    # Maximum drawdown
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    portfolio_running_max = portfolio_cumulative.cummax()
    benchmark_running_max = benchmark_cumulative.cummax()
    
    portfolio_drawdown = (portfolio_cumulative - portfolio_running_max) / portfolio_running_max
    benchmark_drawdown = (benchmark_cumulative - benchmark_running_max) / benchmark_running_max
    
    portfolio_max_drawdown = portfolio_drawdown.min() if len(portfolio_drawdown) > 0 else 0
    benchmark_max_drawdown = benchmark_drawdown.min() if len(benchmark_drawdown) > 0 else 0
    
    # Compile results
    performance_metrics = {
        'portfolio_annualized_return': portfolio_annualized_return * 100,  # Convert to percentage
        'benchmark_annualized_return': benchmark_annualized_return * 100,  # Convert to percentage
        'portfolio_volatility': portfolio_volatility * 100,  # Convert to percentage
        'benchmark_volatility': benchmark_volatility * 100,  # Convert to percentage
        'portfolio_sharpe': portfolio_sharpe,
        'benchmark_sharpe': benchmark_sharpe,
        'portfolio_max_drawdown': portfolio_max_drawdown * 100,  # Convert to percentage
        'benchmark_max_drawdown': benchmark_max_drawdown * 100,  # Convert to percentage
    }
    
    # Create derived arrays expected by the app
    if not results_df.empty:
        base_port = results_df['portfolio_value'].iloc[0]
        base_bench = results_df['benchmark_value'].iloc[0]
        portfolio_cum_return = ((results_df['portfolio_value'] / base_port) - 1) * 100
        benchmark_cum_return = ((results_df['benchmark_value'] / base_bench) - 1) * 100
    else:
        portfolio_cum_return = pd.Series([], dtype=float)
        benchmark_cum_return = pd.Series([], dtype=float)

    return {
        'results_df': results_df,
        'performance_metrics': performance_metrics,
        'rebalance_dates': rebalance_dates,
        'portfolio_holdings': portfolio_holdings,
        # App-friendly keys
        'dates': results_df['date'].tolist(),
        'portfolio_cum_return': portfolio_cum_return.tolist(),
        'benchmark_cum_return': benchmark_cum_return.tolist(),
        'portfolio_return': performance_metrics['portfolio_annualized_return'],
        'benchmark_return': performance_metrics['benchmark_annualized_return'],
        'portfolio_volatility': performance_metrics['portfolio_volatility'],
        'portfolio_sharpe': performance_metrics['portfolio_sharpe'],
        'benchmark_sharpe': performance_metrics['benchmark_sharpe']
    }

def plot_backtest_results(backtest_results, title="Backtest Performance"):
    """
    Plot the results of a backtest.
    
    Parameters:
    -----------
    backtest_results : dict
        Dictionary containing backtest results from run_backtest()
    title : str, optional (default="Backtest Performance")
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    results_df = backtest_results['results_df']
    performance_metrics = backtest_results['performance_metrics']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot portfolio and benchmark values
    ax.plot(results_df['date'], results_df['portfolio_value'], label='Portfolio')
    ax.plot(results_df['date'], results_df['benchmark_value'], label='Benchmark', linestyle='--')
    
    # Add vertical lines for rebalancing dates
    for date in backtest_results['rebalance_dates']:
        ax.axvline(x=date, color='gray', linestyle=':', alpha=0.5)
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title)
    
    # Add performance metrics as text
    text_str = f"Portfolio: {performance_metrics['portfolio_annualized_return']:.2f}% return, "
    text_str += f"{performance_metrics['portfolio_volatility']:.2f}% vol, "
    text_str += f"Sharpe: {performance_metrics['portfolio_sharpe']:.2f}, "
    text_str += f"Max DD: {performance_metrics['portfolio_max_drawdown']:.2f}%\n"
    text_str += f"Benchmark: {performance_metrics['benchmark_annualized_return']:.2f}% return, "
    text_str += f"{performance_metrics['benchmark_volatility']:.2f}% vol, "
    text_str += f"Sharpe: {performance_metrics['benchmark_sharpe']:.2f}, "
    text_str += f"Max DD: {performance_metrics['benchmark_max_drawdown']:.2f}%"
    
    # Position the text box in figure coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def generate_backtest_report(backtest_results):
    """
    Generate a textual report of backtest results.
    
    Parameters:
    -----------
    backtest_results : dict
        Dictionary containing backtest results from run_backtest()
        
    Returns:
    --------
    str
        Textual report
    """
    performance_metrics = backtest_results['performance_metrics']
    rebalance_dates = backtest_results['rebalance_dates']
    portfolio_holdings = backtest_results['portfolio_holdings']
    
    # Format dates
    date_format = "%Y-%m-%d"
    formatted_dates = [date.strftime(date_format) for date in rebalance_dates]
    
    # Create report
    report = "# Backtest Results\n\n"
    
    # Performance summary
    report += "## Performance Summary\n\n"
    report += f"- **Start Date:** {backtest_results['results_df']['date'].iloc[0].strftime(date_format)}\n"
    report += f"- **End Date:** {backtest_results['results_df']['date'].iloc[-1].strftime(date_format)}\n"
    report += f"- **Initial Investment:** ${backtest_results['results_df']['portfolio_value'].iloc[0]:,.2f}\n"
    report += f"- **Final Portfolio Value:** ${backtest_results['results_df']['portfolio_value'].iloc[-1]:,.2f}\n"
    report += f"- **Total Return:** {(backtest_results['results_df']['portfolio_value'].iloc[-1] / backtest_results['results_df']['portfolio_value'].iloc[0] - 1) * 100:.2f}%\n"
    report += f"- **Benchmark Final Value:** ${backtest_results['results_df']['benchmark_value'].iloc[-1]:,.2f}\n"
    report += f"- **Benchmark Total Return:** {(backtest_results['results_df']['benchmark_value'].iloc[-1] / backtest_results['results_df']['benchmark_value'].iloc[0] - 1) * 100:.2f}%\n\n"
    
    # Annualized metrics
    report += "## Annualized Metrics\n\n"
    report += "| Metric | Portfolio | Benchmark |\n"
    report += "|--------|-----------|-----------|\n"
    report += f"| Return | {performance_metrics['portfolio_annualized_return']:.2f}% | {performance_metrics['benchmark_annualized_return']:.2f}% |\n"
    report += f"| Volatility | {performance_metrics['portfolio_volatility']:.2f}% | {performance_metrics['benchmark_volatility']:.2f}% |\n"
    report += f"| Sharpe Ratio | {performance_metrics['portfolio_sharpe']:.2f} | {performance_metrics['benchmark_sharpe']:.2f} |\n"
    report += f"| Max Drawdown | {performance_metrics['portfolio_max_drawdown']:.2f}% | {performance_metrics['benchmark_max_drawdown']:.2f}% |\n\n"
    
    # Rebalancing history
    report += "## Rebalancing History\n\n"
    for i, (date, holdings) in enumerate(zip(formatted_dates, portfolio_holdings)):
        report += f"### Rebalance {i+1}: {date}\n\n"
        report += "| Fund | Allocation |\n"
        report += "|------|------------|\n"
        for ticker, allocation in holdings.items():
            report += f"| {ticker} | ${allocation:,.2f} |\n"
        report += "\n"
    
    # Conclusion
    report += "## Conclusion\n\n"
    
    # Compare portfolio to benchmark
    if performance_metrics['portfolio_annualized_return'] > performance_metrics['benchmark_annualized_return']:
        outperformance = performance_metrics['portfolio_annualized_return'] - performance_metrics['benchmark_annualized_return']
        report += f"The portfolio **outperformed** the benchmark by {outperformance:.2f}% annually. "
    else:
        underperformance = performance_metrics['benchmark_annualized_return'] - performance_metrics['portfolio_annualized_return']
        report += f"The portfolio **underperformed** the benchmark by {underperformance:.2f}% annually. "
    
    # Comment on risk-adjusted returns
    if performance_metrics['portfolio_sharpe'] > performance_metrics['benchmark_sharpe']:
        report += f"On a risk-adjusted basis, the portfolio achieved a higher Sharpe ratio ({performance_metrics['portfolio_sharpe']:.2f} vs. {performance_metrics['benchmark_sharpe']:.2f}), "
        report += "indicating better returns per unit of risk.\n\n"
    else:
        report += f"On a risk-adjusted basis, the portfolio achieved a lower Sharpe ratio ({performance_metrics['portfolio_sharpe']:.2f} vs. {performance_metrics['benchmark_sharpe']:.2f}), "
        report += "indicating worse returns per unit of risk.\n\n"
    
    # Comment on drawdowns
    if performance_metrics['portfolio_max_drawdown'] < performance_metrics['benchmark_max_drawdown']:
        report += f"The portfolio experienced smaller maximum drawdowns ({performance_metrics['portfolio_max_drawdown']:.2f}% vs. {performance_metrics['benchmark_max_drawdown']:.2f}%), "
        report += "suggesting better downside protection.\n\n"
    else:
        report += f"The portfolio experienced larger maximum drawdowns ({performance_metrics['portfolio_max_drawdown']:.2f}% vs. {performance_metrics['benchmark_max_drawdown']:.2f}%), "
        report += "suggesting worse downside protection.\n\n"
    
    # Final assessment
    if (performance_metrics['portfolio_annualized_return'] > performance_metrics['benchmark_annualized_return'] and 
        performance_metrics['portfolio_sharpe'] > performance_metrics['benchmark_sharpe']):
        report += "Overall, the fund ranking strategy demonstrated **strong performance**, delivering both higher returns and better risk-adjusted metrics than the benchmark."
    elif performance_metrics['portfolio_annualized_return'] > performance_metrics['benchmark_annualized_return']:
        report += "Overall, the fund ranking strategy delivered **higher returns** than the benchmark, but with increased risk."
    elif performance_metrics['portfolio_sharpe'] > performance_metrics['benchmark_sharpe']:
        report += "Overall, the fund ranking strategy delivered **better risk-adjusted returns** than the benchmark, but with lower absolute returns."
    else:
        report += "Overall, the fund ranking strategy **underperformed** the benchmark in both absolute and risk-adjusted terms."
    
    return report