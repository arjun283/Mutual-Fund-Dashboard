import os
import subprocess
import sys

def check_dependencies():
    """
    Check if required dependencies are installed.
    """
    try:
        import streamlit
        import pandas
        import numpy
        import matplotlib
        import yfinance
        import statsmodels
        print("All required dependencies are installed.")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required dependencies using: pip install -r requirements.txt")
        return False

def generate_sample_data():
    """
    Generate sample data for offline testing.
    """
    try:
        print("Generating sample data...")
        from sample_data_generator import save_sample_data, generate_combined_csv
        
        # Create sample_data directory if it doesn't exist
        os.makedirs("sample_data", exist_ok=True)
        
        # Generate sample data
        save_sample_data(num_funds=10)
        generate_combined_csv(num_funds=10)
        
        print("Sample data generated successfully.")
        return True
    except Exception as e:
        print(f"Error generating sample data: {e}")
        return False

def run_streamlit():
    """
    Run the Streamlit application.
    """
    try:
        print("\nStarting Streamlit application...")
        print("\nAccess the application in your web browser at: http://localhost:8501")
        print("\nPress Ctrl+C to stop the application.")
        
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
        return True
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        return False

def main():
    print("=== Mutual Fund Recommendation System Example ===\n")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Generate sample data
    if not generate_sample_data():
        return
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main()