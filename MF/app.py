import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import base64
from datetime import datetime, timedelta

# Import custom modules
import data
import metrics
import model
import utils
import backtest

# Default currency symbol for sections that render before the sidebar sets currency
currency_symbol = "₹"

# SIP calculation helper at module level so it is available across reruns
def calculate_sip(monthly_investment, expected_return, time_period_years):
    # Convert annual return to monthly
    monthly_rate = expected_return / 12 / 100
    # Number of months
    months = time_period_years * 12

    # Calculate future value
    future_value = monthly_investment * ((pow(1 + monthly_rate, months) - 1) / monthly_rate) * (1 + monthly_rate)

    # Calculate total investment
    total_investment = monthly_investment * months

    # Calculate wealth gained
    wealth_gained = future_value - total_investment

    return {
        'future_value': future_value,
        'total_investment': total_investment,
        'wealth_gained': wealth_gained
    }

# Set page configuration
st.set_page_config(page_title="Mutual Fund Recommendation System", layout="wide")

# App title and description
st.title("Mutual Fund Recommendation System")
st.markdown("""
This application helps you find the best mutual funds based on your risk tolerance, 
investment amount, and time horizon. It analyzes historical performance and provides 
recommendations tailored to your preferences.
""")

# Create tabs for main app and SIP calculator
tab1, tab2 = st.tabs(["Fund Recommendations", "SIP Calculator"])

with tab2:
    st.header("SIP Calculator")
    st.write("Calculate the future value of your Systematic Investment Plan (SIP)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SIP Calculator inputs
        monthly_investment = st.number_input(
            "Monthly Investment Amount",
            min_value=100,
            max_value=1000000,
            value=5000,
            step=100,
            help="How much you plan to invest monthly"
        )
        
        expected_return = st.slider(
            "Expected Annual Return (%)",
            min_value=1.0,
            max_value=30.0,
            value=12.0,
            step=0.5,
            help="Expected annual return rate"
        )
        
        time_period_years = st.slider(
            "Investment Time Period (Years)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="How long you plan to continue the SIP"
        )
        
        # calculate_sip is defined at module level
        
        # Initialize session state flag for SIP chart visibility
        if 'show_sip_chart' not in st.session_state:
            st.session_state['show_sip_chart'] = False

        # Calculate SIP results
        if st.button("Calculate SIP Returns"):
            results = calculate_sip(monthly_investment, expected_return, time_period_years)

            # Store results in session state for visualization
            st.session_state.sip_results = results
            st.session_state.sip_monthly_investment = monthly_investment
            st.session_state.sip_expected_return = expected_return
            st.session_state.sip_time_period_years = time_period_years
            # Set flag to show the chart on rerun
            st.session_state['show_sip_chart'] = True

            # Display results (metrics + explicit formatted numbers to avoid clipping)
            st.subheader("SIP Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Investment", f"{currency_symbol}{results['total_investment']:,.2f}")
                # explicit formatted text below metric to avoid truncation in narrow layouts
                st.markdown(f"**{currency_symbol}{results['total_investment']:,.2f}**")

            with col2:
                st.metric(
                    "Estimated Returns",
                    f"{currency_symbol}{results['wealth_gained']:,.2f}",
                    delta=f"{(results['wealth_gained']/results['total_investment']*100):.1f}%"
                )
                st.markdown(f"**{currency_symbol}{results['wealth_gained']:,.2f}**")

            with col3:
                st.metric("Future Value", f"{currency_symbol}{results['future_value']:,.2f}")
                st.markdown(f"**{currency_symbol}{results['future_value']:,.2f}**")
    
    with col2:
        # Create data for visualization
        # Show the chart if results exist and the user requested it
        if 'sip_results' in st.session_state and st.session_state.get('show_sip_chart', False):
            # Get values from session state
            monthly_investment = st.session_state.sip_monthly_investment
            expected_return = st.session_state.sip_expected_return
            time_period_years = st.session_state.sip_time_period_years

            # Create year-by-year breakdown
            years = list(range(1, time_period_years + 1))
            invested_amounts = [monthly_investment * 12 * year for year in years]

            # Calculate cumulative values and returns for each year
            cumulative_values = []
            yearly_returns = []
            for year in years:
                result = calculate_sip(monthly_investment, expected_return, year)
                cumulative_values.append(result['future_value'])
                yearly_returns.append(result['future_value'] - (monthly_investment * 12 * year))

            # Create DataFrame for plotting
            df_plot = pd.DataFrame({
                'Year': years,
                'Invested Amount': invested_amounts,
                'Expected Returns': yearly_returns,
                'Total Value': cumulative_values
            })

            # Use plotly to create stacked bars + total line for reliable rendering
            fig = px.bar(
                df_plot,
                x='Year',
                y=['Invested Amount', 'Expected Returns'],
                labels={'value': 'Amount', 'variable': 'Component'},
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )

            fig.add_trace(
                go.Scatter(
                    x=df_plot['Year'],
                    y=df_plot['Total Value'],
                    mode='lines+markers',
                    name='Total Value',
                    line=dict(color='green', width=3)
                )
            )

            fig.update_layout(
                title='Year-by-Year SIP Growth',
                xaxis_title='Year',
                yaxis_title=f'Amount ({currency_symbol})',
                legend_title='Component',
                hovermode='x unified',
                margin=dict(l=40, r=20, t=60, b=40)
            )

            # Show the underlying DataFrame for debugging/visibility
            st.dataframe(df_plot)

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.info("Enter your SIP details and click 'Calculate SIP Returns' to see the visualization")

with tab1:
    # Fund recommendation content goes here
    pass

# Sidebar for user inputs
with st.sidebar:
    st.header("Your Investment Preferences")
    
    # Risk tolerance selection
    risk_tolerance = st.selectbox(
        "Risk Tolerance",
        options=["low", "medium", "high"],
        help="Low: Conservative approach, Medium: Balanced approach, High: Aggressive approach"
    )
    
    # Currency selection
    currency = st.selectbox("Currency", options=["INR", "USD"], index=0)
    currency_symbol = "₹" if currency == "INR" else "$"
    
    # Investment amount
    invest_amount = st.number_input(
        f"Investment Amount ({currency_symbol})",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=1000,
        help="Total amount you want to invest"
    )
    
    # Time horizon
    time_horizon_years = st.slider(
        "Time Horizon (Years)",
        min_value=1,
        max_value=10,
        value=3,
        help="How long you plan to stay invested"
    )
    
    # Number of funds to recommend
    top_n = st.slider(
        "Number of Funds to Recommend",
        min_value=1,
        max_value=10,
        value=5,
        help="How many top funds you want to see"
    )
    
    # Analysis period
    analysis_period_years = st.slider(
        "Analysis Period (Years)",
        min_value=1,
        max_value=5,
        value=3,
        help="How many years of historical data to analyze"
    )
    
    # Benchmark selection
    benchmark_options = {
        "INR": ["NIFTY 50", "SENSEX", "NIFTY 500"],
        "USD": ["S&P 500", "NASDAQ Composite", "Dow Jones Industrial Average"]
    }
    benchmark = st.selectbox(
        "Benchmark Index",
        options=benchmark_options[currency],
        index=0,
        help="Market index to compare fund performance against"
    )
    
    # Risk-free rate
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=1.0,
        max_value=10.0,
        value=4.0 if currency == "INR" else 2.0,
        step=0.1,
        help="Current risk-free interest rate in the market"
    ) / 100.0

# Data source selection
st.header("Data Source")
data_source = st.radio(
    "Select Data Source",
    options=["Live Data (yfinance)", "Upload CSV", "Use Sample Data"],
    index=2,  # Default to sample data
    help="Choose where to get mutual fund data from"
)

# Handle data source selection
df_funds = None
if data_source == "Live Data (yfinance)":
    st.info("Enter mutual fund tickers separated by commas (e.g., HDFC.BO, ICICI.BO)")
    tickers_input = st.text_input("Fund Tickers", "")
    
    if tickers_input:
        tickers = [ticker.strip() for ticker in tickers_input.split(',')]
        with st.spinner("Fetching data from yfinance..."):
            try:
                df_funds = data.fetch_live_data(tickers, analysis_period_years)
                st.success(f"Successfully fetched data for {len(tickers)} funds")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

elif data_source == "Upload CSV":
    st.info("Upload a CSV file with columns: date, ticker, nav")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_funds = data.process_uploaded_csv(uploaded_file)
            st.success(f"Successfully loaded data with {df_funds['ticker'].nunique()} funds")
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

else:  # Use Sample Data
    with st.spinner("Generating sample data..."):
        df_funds = data.generate_sample_data(analysis_period_years, currency)
        st.success(f"Generated sample data with {df_funds['ticker'].nunique()} funds")

# Download sample data template
if st.button("Download Sample CSV Template"):
    sample_df = data.generate_sample_data(1, currency, num_funds=3)
    csv = sample_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_mutual_funds.csv">Download Sample CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

# Process data and generate recommendations if data is available
if df_funds is not None:
    st.header("Fund Analysis and Recommendations")
    
    # Get benchmark data
    benchmark_ticker = utils.get_benchmark_ticker(benchmark, currency)
    df_benchmark = data.get_benchmark_data(benchmark_ticker, analysis_period_years)
    
    # Calculate metrics for all funds
    with st.spinner("Calculating fund metrics..."):
        funds_metrics = metrics.calculate_all_metrics(
            df_funds, 
            df_benchmark, 
            risk_free_rate,
            analysis_period_years
        )
    
    # Rank funds based on user preferences
    with st.spinner("Ranking funds based on your preferences..."):
        ranked_funds = model.rank_funds(
            funds_metrics, 
            risk_tolerance, 
            top_n
        )
    
    # Display ranked funds table
    if not ranked_funds.empty:
        st.subheader(f"Top {top_n} Recommended Funds")
        
        # Format the ranked funds table for display
        display_cols = [
            "Fund Name", "Composite Score", "Annualized Return (%)", 
            "Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)", 
            "Sortino Ratio", "Beta"
        ]
        
        display_df = ranked_funds[display_cols].copy()
        display_df["Annualized Return (%)"] = display_df["Annualized Return (%)"].round(2)
        display_df["Volatility (%)"] = display_df["Volatility (%)"].round(2)
        display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].round(2)
        display_df["Max Drawdown (%)"] = display_df["Max Drawdown (%)"].round(2)
        display_df["Sortino Ratio"] = display_df["Sortino Ratio"].round(2)
        display_df["Beta"] = display_df["Beta"].round(2)
        display_df["Composite Score"] = display_df["Composite Score"].round(2)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Generate allocation suggestion
        st.subheader("Suggested Allocation")
        allocation = model.suggest_allocation(
            ranked_funds, 
            invest_amount, 
            risk_tolerance
        )
        
        # Display allocation as a pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(
            allocation["Amount"], 
            labels=allocation["Fund Name"], 
            autopct='%1.1f%%',
            startangle=90
        )
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        st.pyplot(fig)
        
        # Format amount for display and create a copy for the dataframe
        display_allocation = allocation.copy()
        display_allocation["Amount"] = display_allocation["Amount"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        st.dataframe(display_allocation[["Fund Name", "Amount", "Weight Display"]], use_container_width=True)
        
        # SIP Schedule
        st.subheader("Systematic Investment Plan (SIP) Schedule")
        monthly_sip = invest_amount / (time_horizon_years * 12)
        st.write(f"Recommended Monthly SIP: {currency_symbol}{monthly_sip:,.2f}")
        
        # Monte Carlo simulation for expected outcomes
        st.subheader("Expected Investment Outcomes")
        st.write("Based on Monte Carlo simulation of historical returns:")
        
        simulation_results = model.simulate_portfolio_outcomes(
            ranked_funds, 
            allocation["Weight"].values, 
            invest_amount,
            time_horizon_years,
            num_simulations=1000
        )
        
        # Display simulation results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(simulation_results["time"], simulation_results["median"], 'b-', label="Median Outcome")
        ax.fill_between(
            simulation_results["time"],
            simulation_results["lower"],
            simulation_results["upper"],
            color='b', alpha=0.2,
            label="80% Confidence Interval"
        )
        ax.set_xlabel("Years")
        ax.set_ylabel(f"Portfolio Value ({currency_symbol})")
        ax.set_title("Projected Portfolio Growth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Display expected returns at the end of time horizon
        final_median = simulation_results["median"][-1]
        final_lower = simulation_results["lower"][-1]
        final_upper = simulation_results["upper"][-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Conservative Estimate", 
                f"{currency_symbol}{final_lower:,.2f}",
                f"{(final_lower/invest_amount - 1)*100:.1f}%"
            )
        with col2:
            st.metric(
                "Expected Value", 
                f"{currency_symbol}{final_median:,.2f}",
                f"{(final_median/invest_amount - 1)*100:.1f}%"
            )
        with col3:
            st.metric(
                "Optimistic Estimate", 
                f"{currency_symbol}{final_upper:,.2f}",
                f"{(final_upper/invest_amount - 1)*100:.1f}%"
            )
        
        # Detailed fund analysis (expandable sections)
        st.subheader("Detailed Fund Analysis")
        for idx, (_, fund) in enumerate(ranked_funds.iterrows()):
            with st.expander(f"{idx+1}. {fund['Fund Name']} (Score: {fund['Composite Score']:.2f})"):
                # Fund details in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Key Metrics:**")
                    metrics_df = pd.DataFrame({
                        "Metric": [
                            "Annualized Return",
                            "Volatility",
                            "Sharpe Ratio",
                            "Sortino Ratio",
                            "Max Drawdown",
                            "Beta",
                            "Alpha",
                            "3-Month Momentum",
                            "6-Month Momentum",
                            "12-Month Momentum"
                        ],
                        "Value": [
                            f"{fund['Annualized Return (%)']:.2f}%",
                            f"{fund['Volatility (%)']:.2f}%",
                            f"{fund['Sharpe Ratio']:.2f}",
                            f"{fund['Sortino Ratio']:.2f}",
                            f"{fund['Max Drawdown (%)']:.2f}%",
                            f"{fund['Beta']:.2f}",
                            f"{fund['Alpha']:.2f}%",
                            f"{fund['3M Momentum']:.2f}%",
                            f"{fund['6M Momentum']:.2f}%",
                            f"{fund['12M Momentum']:.2f}%"
                        ]
                    })
                    st.table(metrics_df)
                
                with col2:
                    # Get fund price history
                    fund_ticker = fund["Ticker"]
                    fund_data = df_funds[df_funds["ticker"] == fund_ticker]
                    
                    # Plot price series
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(fund_data["date"], fund_data["nav"], label=fund["Fund Name"])
                    ax.set_xlabel("Date")
                    ax.set_ylabel("NAV")
                    ax.set_title(f"{fund['Fund Name']} - Price History")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Cumulative return comparison with benchmark
                st.write("**Cumulative Return Comparison:**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Calculate cumulative returns
                fund_data = fund_data.sort_values("date")
                fund_cum_return = (fund_data["nav"] / fund_data["nav"].iloc[0] - 1) * 100
                
                benchmark_data = df_benchmark.sort_values("date")
                benchmark_cum_return = (benchmark_data["nav"] / benchmark_data["nav"].iloc[0] - 1) * 100
                
                # Plot cumulative returns
                ax.plot(fund_data["date"], fund_cum_return, label=fund["Fund Name"])
                ax.plot(benchmark_data["date"], benchmark_cum_return, label=benchmark)
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative Return (%)")
                ax.set_title("Cumulative Return Comparison")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Rolling volatility
                st.write("**Rolling Volatility (30-day window):**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Calculate rolling volatility
                rolling_vol = metrics.calculate_rolling_volatility(fund_data, window=30)
                
                # Plot rolling volatility
                ax.plot(rolling_vol.index, rolling_vol.values * 100)
                ax.set_xlabel("Date")
                ax.set_ylabel("Annualized Volatility (%)")
                ax.set_title(f"{fund['Fund Name']} - Rolling Volatility")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # CAPM regression plot
                st.write("**CAPM Regression:**")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get CAPM regression data
                capm_data = metrics.get_capm_regression_data(fund_data, benchmark_data)
                
                # Plot CAPM regression
                ax.scatter(capm_data["benchmark_return"] * 100, capm_data["fund_return"] * 100, alpha=0.5)
                
                # Add regression line
                beta = fund["Beta"]
                alpha = fund["Alpha"]
                x_range = np.linspace(capm_data["benchmark_return"].min() * 100, capm_data["benchmark_return"].max() * 100, 100)
                y_range = alpha + beta * x_range
                ax.plot(x_range, y_range, 'r-', label=f"Alpha: {alpha:.2f}%, Beta: {beta:.2f}")
                
                ax.set_xlabel("Benchmark Return (%)")
                ax.set_ylabel("Fund Return (%)")
                ax.set_title("CAPM Regression")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Textual interpretation
                st.write("**Interpretation:**")
                interpretation = utils.generate_fund_interpretation(
                    fund, 
                    risk_tolerance, 
                    benchmark
                )
                st.write(interpretation)
        
        # Risk-Return Scatter Plot
        st.subheader("Risk-Return Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all funds
        ax.scatter(
            ranked_funds["Volatility (%)"],
            ranked_funds["Annualized Return (%)"],
            s=100,
            alpha=0.7,
            label="Recommended Funds"
        )
        
        # Add benchmark
        benchmark_metrics = metrics.calculate_benchmark_metrics(df_benchmark, risk_free_rate)
        ax.scatter(
            benchmark_metrics["Volatility (%)"],
            benchmark_metrics["Annualized Return (%)"],
            s=150,
            marker="*",
            color="red",
            label=benchmark
        )
        
        # Add labels for each fund
        for idx, fund in ranked_funds.iterrows():
            ax.annotate(
                fund["Fund Name"],
                (fund["Volatility (%)"], fund["Annualized Return (%)"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8
            )
        
        # Add efficient frontier line
        if len(ranked_funds) >= 3:
            try:
                x_range = np.linspace(
                    ranked_funds["Volatility (%)"].min() * 0.8,
                    ranked_funds["Volatility (%)"].max() * 1.2,
                    100
                )
                y_range = model.efficient_frontier_approximation(
                    ranked_funds["Volatility (%)"].values,
                    ranked_funds["Annualized Return (%)"].values,
                    x_range
                )
                ax.plot(x_range, y_range, 'g--', label="Efficient Frontier Approximation")
            except Exception as e:
                st.warning(f"Could not generate efficient frontier: {str(e)}")
        
        ax.set_xlabel("Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.set_title("Risk-Return Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Performance comparison chart
        st.subheader("Performance Comparison")
        st.write("Comparing the performance of recommended funds against the benchmark:")
        
        # Get historical data for the top funds
        top_fund_tickers = ranked_funds['Ticker'].tolist()
        top_fund_data = df_funds[df_funds['ticker'].isin(top_fund_tickers)].copy()
        
        # Prepare data for plotting
        pivot_data = top_fund_data.pivot(index='date', columns='ticker', values='nav')
        
        # Normalize to 100 at the start
        normalized_data = pivot_data / pivot_data.iloc[0] * 100
        
        # Add benchmark to the plot
        benchmark_data = df_benchmark.copy()
        benchmark_data = benchmark_data.set_index('date')
        normalized_data[benchmark] = benchmark_data['nav'] / benchmark_data['nav'].iloc[0] * 100
        
        # Get Nifty and Sensex data if not already included
        additional_benchmarks = []
        if benchmark_ticker != 'NSEI.NS' and currency == 'INR':
            additional_benchmarks.append('NSEI.NS')  # Nifty 50
        if benchmark_ticker != 'BSESN.NS' and currency == 'INR':
            additional_benchmarks.append('BSESN.NS')  # Sensex
            
        if additional_benchmarks:
            with st.spinner("Fetching additional benchmark data..."):
                for ticker in additional_benchmarks:
                    try:
                        bench_name = "NIFTY 50" if ticker == "NSEI.NS" else "SENSEX"
                        add_bench_data = data.get_benchmark_data(ticker, analysis_period_years)
                        add_bench_data = add_bench_data.set_index('date')
                        normalized_data[bench_name] = add_bench_data['nav'] / add_bench_data['nav'].iloc[0] * 100
                    except Exception as e:
                        st.warning(f"Could not fetch data for {bench_name}: {str(e)}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        normalized_data.plot(ax=ax)
        ax.set_title(f"Fund Performance vs Benchmarks (Normalized to 100)")
        ax.set_ylabel("Normalized Value")
        ax.set_xlabel("Date")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        
        # Add trend lines for each series
        for col in normalized_data.columns:
            # Create a copy of the data without NaN values
            clean_data = normalized_data[col].dropna()
            if len(clean_data) > 1:  # Need at least 2 points for a trendline
                # Convert dates to numeric for regression
                x_numeric = mdates.date2num(clean_data.index)
                # Fit a linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, clean_data)
                # Create trendline
                trendline_x = [clean_data.index[0], clean_data.index[-1]]
                trendline_y = [slope * mdates.date2num(trendline_x[0]) + intercept,
                              slope * mdates.date2num(trendline_x[-1]) + intercept]
                # Add trendline to plot
                ax.plot(trendline_x, trendline_y, '--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Backtest results
        st.subheader("Backtest Results")
        st.write("Performance of the ranking strategy over historical data:")
        
        with st.spinner("Running backtest..."):
            # Calculate start and end dates for backtest
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year lookback
            
            backtest_results = backtest.run_backtest(
                df_funds,
                df_benchmark,
                start_date,
                end_date,
                rebalance_freq='quarterly',  # Equivalent to 3 months
                risk_tolerance=risk_tolerance,
                top_n=top_n
            )
        
        # Plot backtest results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            backtest_results["dates"],
            backtest_results["portfolio_cum_return"],
            label="Strategy Portfolio"
        )
        ax.plot(
            backtest_results["dates"],
            backtest_results["benchmark_cum_return"],
            label=benchmark
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_title("Backtest Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Backtest metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Strategy Return", 
                f"{backtest_results['portfolio_return']:.2f}%",
                f"{backtest_results['portfolio_return'] - backtest_results['benchmark_return']:.2f}%"
            )
        with col2:
            st.metric(
                "Benchmark Return", 
                f"{backtest_results['benchmark_return']:.2f}%"
            )
        with col3:
            st.metric(
                "Strategy Volatility", 
                f"{backtest_results['portfolio_volatility']:.2f}%"
            )
        with col4:
            st.metric(
                "Strategy Sharpe", 
                f"{backtest_results['portfolio_sharpe']:.2f}",
                f"{backtest_results['portfolio_sharpe'] - backtest_results['benchmark_sharpe']:.2f}"
            )
        
        # Export options
        st.subheader("Export Recommendations")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as CSV"):
                # Prepare export data
                export_df = pd.concat([
                    ranked_funds,
                    allocation.set_index("Fund Name")
                ], axis=1, join="inner")
                
                # Convert to CSV
                csv = export_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="fund_recommendations.csv">Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            if st.button("Export as PDF Summary"):
                st.warning("PDF export functionality requires additional setup. Please use CSV export instead.")
        
        # Discussion notes
        with st.expander("Discussion Notes"):
            st.write("""
            ### Methodology
            
            The fund ranking methodology uses a composite scoring system that weights various metrics based on the user's risk tolerance:
            
            - **Low Risk**: Emphasizes downside protection metrics like Sortino ratio, maximum drawdown, and beta.
            - **Medium Risk**: Balances return and risk metrics for a moderate approach.
            - **High Risk**: Prioritizes return potential and momentum indicators for aggressive growth.
            
            ### Limitations
            
            1. **Past Performance Disclaimer**: Historical performance does not guarantee future results.
            2. **Data Limitations**: The analysis is limited by the quality and availability of historical data.
            3. **Model Simplicity**: The ranking model uses simplified weights and does not account for all market factors.
            4. **Market Conditions**: The recommendations may perform differently under changing market conditions.
            
            ### Ethical Considerations
            
            1. **Transparency**: The methodology is transparent with no black-box algorithms.
            2. **Risk Disclosure**: The system clearly communicates the risks associated with investments.
            3. **No Guarantees**: The system does not promise specific returns or outcomes.
            4. **Educational Purpose**: This tool is designed for educational purposes and should not replace professional financial advice.
            
            ### Possible Improvements
            
            1. **Factor Models**: Incorporate multi-factor models for more sophisticated analysis.
            2. **Machine Learning**: Implement more advanced ML techniques with proper validation.
            3. **Economic Indicators**: Include macroeconomic indicators for better context.
            4. **Sentiment Analysis**: Integrate more sophisticated market sentiment analysis.
            5. **Portfolio Optimization**: Implement formal portfolio optimization techniques.
            """)

# Footer
st.markdown("---")
st.markdown("""
<center>Mutual Fund Recommendation System | University Project | Not Financial Advice</center>
""", unsafe_allow_html=True)