import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Dictionary of benchmark tickers
BENCHMARK_TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTY 500": "^CRSLDX",
    "S&P 500": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "Dow Jones Industrial Average": "^DJI"
}

def get_benchmark_ticker(benchmark_name, currency="INR"):
    """
    Get the ticker symbol for a benchmark index.
    
    Parameters:
    -----------
    benchmark_name : str
        Name of the benchmark index
    currency : str, optional (default="INR")
        Currency ("INR" or "USD")
        
    Returns:
    --------
    str
        Ticker symbol for the benchmark
    """
    if benchmark_name in BENCHMARK_TICKERS:
        return BENCHMARK_TICKERS[benchmark_name]
    else:
        # Default benchmarks
        if currency == "INR":
            return BENCHMARK_TICKERS["NIFTY 50"]
        else:
            return BENCHMARK_TICKERS["S&P 500"]

def format_currency(amount, currency="INR"):
    """
    Format a currency amount with the appropriate symbol.
    
    Parameters:
    -----------
    amount : float
        Amount to format
    currency : str, optional (default="INR")
        Currency ("INR" or "USD")
        
    Returns:
    --------
    str
        Formatted currency string
    """
    if currency == "INR":
        return f"₹{amount:,.2f}"
    else:
        return f"${amount:,.2f}"

def generate_fund_interpretation(fund, risk_tolerance, benchmark):
    """
    Generate a textual interpretation of a fund's performance and characteristics.
    
    Parameters:
    -----------
    fund : pandas.Series
        Series containing fund metrics
    risk_tolerance : str
        Risk tolerance level ('low', 'medium', or 'high')
    benchmark : str
        Name of the benchmark index
        
    Returns:
    --------
    str
        Textual interpretation
    """
    # Extract key metrics
    fund_name = fund["Fund Name"]
    annualized_return = fund["Annualized Return (%)"]
    volatility = fund["Volatility (%)"]
    sharpe = fund["Sharpe Ratio"]
    sortino = fund["Sortino Ratio"]
    max_drawdown = fund["Max Drawdown (%)"]
    beta = fund["Beta"]
    alpha = fund["Alpha"]
    momentum_3m = fund["3M Momentum"]
    momentum_12m = fund["12M Momentum"]
    
    # Interpret return
    if annualized_return > 15:
        return_desc = "excellent"
    elif annualized_return > 10:
        return_desc = "strong"
    elif annualized_return > 5:
        return_desc = "moderate"
    else:
        return_desc = "modest"
    
    # Interpret risk
    if volatility > 20:
        risk_desc = "high"
    elif volatility > 12:
        risk_desc = "moderate"
    else:
        risk_desc = "low"
    
    # Interpret Sharpe ratio
    if sharpe > 1.5:
        sharpe_desc = "excellent"
    elif sharpe > 1.0:
        sharpe_desc = "good"
    elif sharpe > 0.5:
        sharpe_desc = "adequate"
    else:
        sharpe_desc = "poor"
    
    # Interpret beta
    if beta > 1.2:
        beta_desc = "significantly more volatile than"
    elif beta > 1.0:
        beta_desc = "slightly more volatile than"
    elif beta > 0.8:
        beta_desc = "about as volatile as"
    else:
        beta_desc = "less volatile than"
    
    # Interpret alpha
    if alpha > 5:
        alpha_desc = "significantly outperformed"
    elif alpha > 2:
        alpha_desc = "outperformed"
    elif alpha > -2:
        alpha_desc = "performed similarly to"
    else:
        alpha_desc = "underperformed"
    
    # Interpret momentum
    if momentum_3m > 5 and momentum_12m > 10:
        momentum_desc = "strong positive"
    elif momentum_3m > 0 and momentum_12m > 0:
        momentum_desc = "positive"
    elif momentum_3m < 0 and momentum_12m < 0:
        momentum_desc = "negative"
    else:
        momentum_desc = "mixed"
    
    # Generate interpretation based on risk tolerance
    if risk_tolerance == "low":
        interpretation = f"{fund_name} has shown {return_desc} returns with {risk_desc} volatility. "
        interpretation += f"The fund has a {sharpe_desc} risk-adjusted return (Sharpe ratio of {sharpe:.2f}). "
        interpretation += f"It is {beta_desc} the {benchmark} (Beta of {beta:.2f}) and has {alpha_desc} "
        interpretation += f"the benchmark (Alpha of {alpha:.2f}%). "
        interpretation += f"The maximum drawdown of {max_drawdown:.2f}% indicates the worst historical decline. "
        interpretation += f"Recent momentum has been {momentum_desc}, with a 3-month return of {momentum_3m:.2f}% "
        interpretation += f"and a 12-month return of {momentum_12m:.2f}%."
        
        if risk_desc == "low" and sharpe_desc in ["good", "excellent"]:
            interpretation += f" This fund appears well-suited for your low risk tolerance, offering "
            interpretation += f"a good balance of returns with lower volatility."
        else:
            interpretation += f" While this fund has some positive characteristics, you may want to "
            interpretation += f"consider its {risk_desc} volatility in the context of your low risk tolerance."
    
    elif risk_tolerance == "medium":
        interpretation = f"{fund_name} has delivered {return_desc} returns with {risk_desc} volatility. "
        interpretation += f"The Sharpe ratio of {sharpe:.2f} indicates {sharpe_desc} risk-adjusted returns. "
        interpretation += f"With a Beta of {beta:.2f}, it is {beta_desc} the {benchmark} and has {alpha_desc} "
        interpretation += f"the benchmark with an Alpha of {alpha:.2f}%. "
        interpretation += f"The fund experienced a maximum drawdown of {max_drawdown:.2f}%. "
        interpretation += f"It has shown {momentum_desc} momentum recently, with returns of {momentum_3m:.2f}% "
        interpretation += f"over 3 months and {momentum_12m:.2f}% over 12 months."
        
        if risk_desc == "moderate" and return_desc in ["strong", "excellent"]:
            interpretation += f" This fund aligns well with your medium risk tolerance, offering "
            interpretation += f"a good balance of risk and return potential."
        else:
            interpretation += f" Consider how this fund's {risk_desc} risk profile fits with your "
            interpretation += f"medium risk tolerance and overall portfolio strategy."
    
    else:  # high risk tolerance
        interpretation = f"{fund_name} has generated {return_desc} returns with {risk_desc} volatility. "
        interpretation += f"It has a {sharpe_desc} Sharpe ratio of {sharpe:.2f} and a Sortino ratio of {sortino:.2f}. "
        interpretation += f"With a Beta of {beta:.2f}, the fund is {beta_desc} the {benchmark} and has {alpha_desc} "
        interpretation += f"the benchmark with an Alpha of {alpha:.2f}%. "
        interpretation += f"The fund has experienced a maximum drawdown of {max_drawdown:.2f}%. "
        interpretation += f"Recent performance shows {momentum_desc} momentum with {momentum_3m:.2f}% return "
        interpretation += f"over 3 months and {momentum_12m:.2f}% over 12 months."
        
        if return_desc in ["strong", "excellent"] and momentum_desc == "strong positive":
            interpretation += f" This fund's strong performance and positive momentum align well with "
            interpretation += f"your high risk tolerance, offering growth potential."
        else:
            interpretation += f" Consider how this fund's {return_desc} returns and {momentum_desc} momentum "
            interpretation += f"align with your high risk tolerance and growth objectives."
    
    return interpretation

def create_comparison_chart(funds_data, benchmark_data, fund_names, title="Fund Comparison"):
    """
    Create a comparison chart of multiple funds against a benchmark.
    
    Parameters:
    -----------
    funds_data : list of pandas.DataFrame
        List of DataFrames containing fund data
    benchmark_data : pandas.DataFrame
        DataFrame containing benchmark data
    fund_names : list of str
        List of fund names
    title : str, optional (default="Fund Comparison")
        Chart title
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot benchmark
    benchmark_data = benchmark_data.sort_values('date')
    benchmark_cum_return = (benchmark_data['nav'] / benchmark_data['nav'].iloc[0] - 1) * 100
    ax.plot(benchmark_data['date'], benchmark_cum_return, 'k--', label="Benchmark")
    
    # Plot each fund
    for i, (fund_data, fund_name) in enumerate(zip(funds_data, fund_names)):
        fund_data = fund_data.sort_values('date')
        fund_cum_return = (fund_data['nav'] / fund_data['nav'].iloc[0] - 1) * 100
        ax.plot(fund_data['date'], fund_cum_return, label=fund_name)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def export_to_pdf(ranked_funds, allocation, simulation_results, benchmark_metrics, filename="fund_recommendation.pdf"):
    """
    Export recommendation results to a PDF file.
    
    Parameters:
    -----------
    ranked_funds : pandas.DataFrame
        DataFrame containing ranked funds
    allocation : pandas.DataFrame
        DataFrame containing allocation suggestion
    simulation_results : dict
        Dictionary with simulation results
    benchmark_metrics : dict
        Dictionary with benchmark metrics
    filename : str, optional (default="fund_recommendation.pdf")
        Name of the output PDF file
        
    Returns:
    --------
    str
        Path to the saved PDF file
    """
    # This is a placeholder function
    # In a real implementation, you would use a library like ReportLab or FPDF to generate a PDF
    # For this project, we'll just return the filename
    return filename

def validate_input(risk_tolerance, invest_amount, time_horizon_years, top_n):
    """
    Validate user input parameters.
    
    Parameters:
    -----------
    risk_tolerance : str
        Risk tolerance level ('low', 'medium', or 'high')
    invest_amount : float
        Investment amount
    time_horizon_years : int
        Investment time horizon in years
    top_n : int
        Number of funds to recommend
        
    Returns:
    --------
    tuple
        (is_valid, error_message)
    """
    # Validate risk tolerance
    if risk_tolerance not in ["low", "medium", "high"]:
        return False, f"Invalid risk tolerance: {risk_tolerance}. Use 'low', 'medium', or 'high'."
    
    # Validate investment amount
    if invest_amount <= 0:
        return False, f"Investment amount must be positive."
    
    # Validate time horizon
    if time_horizon_years < 1 or time_horizon_years > 10:
        return False, f"Time horizon must be between 1 and 10 years."
    
    # Validate top_n
    if top_n < 1 or top_n > 10:
        return False, f"Number of funds must be between 1 and 10."
    
    return True, ""