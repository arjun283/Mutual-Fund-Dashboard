# Mutual Fund Recommendation System

A university-level Python project that recommends mutual funds based on user preferences such as risk tolerance, investment amount, and time horizon.

## Features

- **Data Modes**: Live fetch using yfinance or upload CSV with historical NAVs
- **Comprehensive Metrics**: Calculates annualized returns, volatility, downside risk, Sortino ratio, maximum drawdown, CAPM beta/alpha, Sharpe ratio, momentum, and more
- **Transparent Ranking**: Uses a weighted scoring system based on user's risk tolerance
- **Allocation Suggestions**: Provides investment allocation for top funds and SIP schedule
- **Visualization**: Interactive charts for fund performance, risk metrics, and comparisons
- **Backtesting**: Evaluates the ranking method over historical data

## Setup and Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main Streamlit application
- `data.py`: Data ingestion module (live fetch and CSV upload)
- `metrics.py`: Financial metrics calculation
- `model.py`: Fund ranking and allocation models
- `backtest.py`: Backtesting functionality
- `utils.py`: Utility functions
- `tests/`: Unit tests for core functions
- `sample_data/`: Sample data for offline testing

## Usage

1. Select your risk tolerance, investment amount, time horizon, and number of funds to recommend
2. Choose between live data fetch or upload your own CSV
3. View the ranked list of recommended funds with key metrics
4. Explore detailed analysis for each fund
5. Check the suggested allocation and SIP schedule
6. Export the recommendation as CSV or PDF

## Offline Mode

The project includes a sample data generator that allows you to run the application without internet access. Use the sample data option in the UI to test the functionality.

## Limitations and Assumptions

- Historical performance does not guarantee future results
- The ranking model uses simplified weights based on risk tolerance
- Limited to mutual funds available through yfinance or provided in CSV
- Market sentiment proxy is a simplified indicator

## License

This project is created for educational purposes.