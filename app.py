import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Updated style configuration
plt.style.use('seaborn-v0_8')  # Fixed style name
pd.set_option('display.max_columns', None)

# List of popular ETFs to analyze
ETF_LIST = [
    'SPY',  # S&P 500
    'QQQ',  # Nasdaq-100
    'DIA',  # Dow Jones
    'IWM',  # Russell 2000
    'GLD',  # Gold
    'TLT',  # 20+ Year Treasury Bonds
    'VTI',  # Total Stock Market
    'VXUS', # Total International Stock
    'BND',  # Total Bond Market
    'VNQ'   # Real Estate
]

def fetch_etf_data(tickers, start_date='2015-01-01', end_date=None):
    """Fetch historical ETF data from Yahoo Finance"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Fetching ETF data from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    return data

def preprocess_data(etf_data):
    """
    Clean and preprocess ETF data
    """
    processed = {}
    
    for ticker in ETF_LIST:
        df = etf_data[ticker].copy()
        
        # Calculate daily returns and moving averages
        df['Daily_Return'] = df['Close'].pct_change()
        df['5_day_MA'] = df['Close'].rolling(5).mean()
        df['20_day_MA'] = df['Close'].rolling(20).mean()
        df['50_day_MA'] = df['Close'].rolling(50).mean()
        
        # Calculate RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Drop NA values
        df.dropna(inplace=True)
        
        # Add target variable (next day's return)
        df['Target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        processed[ticker] = df
    
    return processed

def train_predictive_models(etf_data):
    """
    Train predictive models for each ETF
    """
    models = {}
    results = {}
    
    for ticker, df in etf_data.items():
        # Features for prediction
        features = ['5_day_MA', '20_day_MA', '50_day_MA', 'RSI', 'MACD', 'Signal_Line', 'Volume']
        X = df[features]
        y = df['Target']
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Create and train model pipeline
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        models[ticker] = model
        results[ticker] = {
            'mse': mse,
            'r2': r2,
            'last_date': df.index[-1],
            'last_price': df['
            
            
            Close'].iloc[-1],
            'predicted_return': y_pred[-1]
        }
    
    return models, results

def generate_recommendations(model_results):
    """
    Generate ETF recommendations based on model predictions
    """
    recommendations = []
    
    # Create DataFrame from results
    results_df = pd.DataFrame.from_dict(model_results, orient='index')
    results_df = results_df.sort_values('predicted_return', ascending=False)
    
    # Generate recommendations
    for ticker, row in results_df.iterrows():
        pred_return = row['predicted_return']
        
        if pred_return > 0.005:  # > 0.5%
            recommendation = "Strong Buy"
        elif pred_return > 0:
            recommendation = "Buy"
        elif pred_return > -0.005:
            recommendation = "Hold"
        else:
            recommendation = "Sell"
        
        recommendations.append({
            'ETF': ticker,
            'Last Price': row['last_price'],
            'Predicted Daily Return': f"{pred_return*100:.2f}%",
            'Recommendation': recommendation,
            'Model R² Score': f"{row['r2']:.2f}",
            'Last Update': row['last_date'].strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(recommendations)

def visualize_etf_performance(etf_data, ticker):
    """
    Visualize technical indicators for an ETF
    """
    df = etf_data[ticker]
    
    plt.figure(figsize=(15, 10))
    
    # Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Price')
    plt.plot(df['5_day_MA'], label='5-day MA')
    plt.plot(df['20_day_MA'], label='20-day MA')
    plt.plot(df['50_day_MA'], label='50-day MA')
    plt.title(f'{ticker} Price and Moving Averages')
    plt.legend()
    
    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['RSI'], label='RSI')
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='g', linestyle='--')
    plt.title('Relative Strength Index (RSI)')
    
    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(df['MACD'], label='MACD')
    plt.plot(df['Signal_Line'], label='Signal Line')
    plt.title('MACD')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Step 1: Fetch ETF data
    etf_data_raw = fetch_etf_data(ETF_LIST)
    
    # Step 2: Preprocess data
    etf_data_processed = preprocess_data(etf_data_raw)
    
    # Step 3: Train predictive models
    models, results = train_predictive_models(etf_data_processed)
    
    # Step 4: Generate recommendations
    recommendations = generate_recommendations(results)
    
    # Display recommendations
    print("\nETF Recommendations:")
    print(recommendations.to_string(index=False))
    
    # Visualize for a specific ETF
    visualize_etf = 'SPY'  # Change to any ETF in ETF_LIST
    visualize_etf_performance(etf_data_processed, visualize_etf)
    
    # Save results to CSV
    recommendations.to_csv('etf_recommendations.csv', index=False)
    print("\nRecommendations saved to 'etf_recommendations.csv'")

if __name__ == "__main__":
    main()
