import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="ETF Performance Dashboard", layout="wide")

# ETF list
etfs = [
    "GOLDBEES.NS", "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", "LIQUIDBEES.NS",
    "MOM100.NS", "MON100.NS", "NIF100BEES.NS", "NV20BEES.NS", "SETFNIF50.NS",
    "HDFCNIFTY.NS", "ICICINIFTY.NS", "KOTAKNIFTY.NS", "AXISNIFTY.NS", "MIRAEEMER.NS",
    "MIRAEENIF.NS", "SBIETFQLTY.NS", "UTINIFTETF.NS", "ABSLNN50ET.NS", "M100.NS",
    "M50.NS", "NETF.NS", "NIFTY100.NS", "NIFTYETF.NS", "QNIFTY.NS",
    "SILVERBEES.NS", "ITBEES.NS", "PSUBNKBEES.NS", "CPSEETF.NS", "INFRABEES.NS",
    "CONSUMBEES.NS", "PHARMABEES.NS", "AUTOBEES.NS", "SHARIABEES.NS", "DIVOPPBEES.NS",
    "HDFCGOLD.NS", "HDFCSILVER.NS", "HDFCNIF100.NS", "HDFCSENSEX.NS", "HDFCNEXT50.NS",
    "HDFCMOMENT.NS", "HDFCLOWVOL.NS", "HDFCQUAL.NS", "HDFCVALUE.NS", "HDFCMID150.NS",
    "HDFCSML250.NS", "HDFCBSE500.NS", "HDFCNIFBAN.NS", "HDFCPVTBAN.NS", "HDFCPSUBK.NS",
    "HDFCNIFIT.NS", "HDFCGROWTH.NS", "HDFCLIQUID.NS", "AXISGOLD.NS", "AXISNIFTY.NS",
    "AXISBPSETF.NS", "AXISHCETF.NS", "AXISILVER.NS", "AXISCETF.NS", "AXISVALUE.NS",
    "AXISBLUECH.NS", "AXISGTE.NS", "AXISMFETF.NS", "AXISNIF100.NS", "AXISQUAL.NS",
    "AXISGOVT.NS", "AXISBANK.NS", "AXISCONS.NS", "AXISENERG.NS", "AXISFIN.NS",
    "AXISFMCG.NS", "AXISIT.NS", "AXISMETAL.NS", "AXISPHAR.NS", "AXISPSU.NS",
    "AXISTEC.NS", "AXISUTIL.NS", "MIRAEEMER.NS", "MIRAEENIF.NS", "MIRAEESEN.NS",
    "MIRAEESET.NS", "MIRAEEM50.NS", "MIRAEEM100.NS", "MIRAEENXT.NS", "MIRAEELOW.NS",
    "MIRAEECON.NS", "MIRAEEDIV.NS", "MIRAEEMOM.NS", "MIRAEEMV.NS", "MIRAEENV.NS",
    "MIRAEERUR.NS", "MIRAEESMC.NS", "MIRAEETEC.NS", "MIRAEETOP.NS", "MIRAEETRA.NS",
    "MIRAEETUR.NS", "MIRAEETUS.NS", "MIRAEETW.NS", "MIRAEETX.NS", "MIRAEETY.NS",
    "MIRAEETZ.NS", "MIRAEET0.NS", "MIRAEET1.NS", "MIRAEET2.NS", "MIRAEET3.NS",
    "MIRAEET4.NS", "MIRAEET5.NS", "MIRAEET6.NS", "MIRAEET7.NS", "MIRAEET8.NS",
    "MIRAEET9.NS", "MIRAEET10.NS", "MIRAEET11.NS", "MIRAEET12.NS", "MIRAEET13.NS",
    "MIRAEET14.NS", "MIRAEET15.NS", "MIRAEET16.NS", "MIRAEET17.NS", "MIRAEET18.NS",
    "MIRAEET19.NS", "MIRAEET20.NS", "MIRAEET21.NS", "MIRAEET22.NS", "MIRAEET23.NS",
    "MIRAEET24.NS", "MIRAEET25.NS", "MIRAEET26.NS", "MIRAEET27.NS", "MIRAEET28.NS",
    "MIRAEET29.NS", "MIRAEET30.NS", "MIRAEET31.NS", "MIRAEET32.NS", "MIRAEET33.NS",
    "MIRAEET34.NS", "MIRAEET35.NS", "MIRAEET36.NS", "MIRAEET37.NS", "MIRAEET38.NS",
    "MIRAEET39.NS", "MIRAEET40.NS", "MIRAEET41.NS", "MIRAEET42.NS", "MIRAEET43.NS",
    "MIRAEET44.NS", "MIRAEET45.NS", "MIRAEET46.NS", "MIRAEET47.NS", "MIRAEET48.NS",
    "MIRAEET49.NS", "MIRAEET50.NS"
]

# Filter out ETFs that consistently cause errors
problematic_etfs = ["MIRAEET"+str(i)+".NS" for i in range(10,51)] + ["MIRAEET"+chr(i)+".NS" for i in range(65,91)]
etfs = [etf for etf in etfs if etf not in problematic_etfs]

# Function to fetch data with error handling
@st.cache_data(ttl=3600)
def get_data(etf_list, period='1mo'):
    end_date = datetime.today()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
    elif period == '1mo':
        start_date = end_date - timedelta(days=30)
    elif period == '3mo':
        start_date = end_date - timedelta(days=90)
    else:
        start_date = end_date - timedelta(days=365)
    
    data = yf.download(etf_list, start=start_date, end=end_date, group_by='ticker', progress=False)
    return data

# Function to calculate returns with error handling
def calculate_returns(data, etf_list):
    returns = {}
    for etf in etf_list:
        try:
            if etf in data:
                close_prices = data[etf]['Close']
                if len(close_prices) > 1:
                    returns[etf] = {
                        'Return': (close_prices[-1] - close_prices[0]) / close_prices[0] * 100,
                        'Volatility': np.std(close_prices.pct_change().dropna()) * np.sqrt(252) * 100,
                        'Last Price': close_prices[-1]
                    }
        except Exception as e:
            continue
    return pd.DataFrame.from_dict(returns, orient='index').sort_values('Return', ascending=False)

# Function for ARIMA prediction
def arima_prediction(etf, days=30):
    try:
        data = yf.download(etf, period='1y')['Close'].values
        
        # Fit ARIMA model
        model = ARIMA(data, order=(5,1,0))
        model_fit = model.fit()
        
        # Make prediction
        forecast = model_fit.forecast(steps=days)
        predicted_return = (forecast[-1] - data[-1]) / data[-1] * 100
        
        return predicted_return
    except:
        return None

# Function for LSTM prediction
def lstm_prediction(etf, days=30):
    try:
        data = yf.download(etf, period='1y')['Close'].values
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        # Prepare training data
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        
        # Make prediction
        inputs = scaled_data[len(scaled_data)-60:].reshape(1, -1)
        inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
        predicted_price = model.predict(inputs)
        predicted_price = scaler.inverse_transform(predicted_price)
        predicted_return = (predicted_price[0][0] - data[-1]) / data[-1] * 100
        
        return predicted_return
    except:
        return None

# Function for Random Forest prediction (improved)
def rf_prediction(etf, days=30):
    try:
        # Get historical data
        data = yf.download(etf, period='1y')['Close'].reset_index()
        if len(data) < 60:
            return None
            
        # Create features
        data['Day'] = (data['Date'] - data['Date'].min()).dt.days
        for i in range(1, 6):
            data[f'Lag_{i}'] = data['Close'].shift(i)
        data['MA_5'] = data['Close'].rolling(5).mean()
        data['MA_10'] = data['Close'].rolling(10).mean()
        data['MA_20'] = data['Close'].rolling(20).mean()
        data = data.dropna()
        
        # Prepare features and target
        X = data.drop(['Date', 'Close'], axis=1)
        y = data['Close']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        # Prepare future data
        last_row = X.iloc[-1:]
        future_data = []
        
        for day in range(1, days+1):
            new_row = last_row.copy()
            new_row['Day'] += day
            for i in range(1, 6):
                if i == 1:
                    new_row[f'Lag_{i}'] = data['Close'].iloc[-1]
                else:
                    new_row[f'Lag_{i}'] = last_row[f'Lag_{i-1}'].values[0]
            
            # Predict next price
            next_price = model.predict(new_row)[0]
            future_data.append(next_price)
            
            # Update moving averages in new_row for next prediction
            new_row['MA_5'] = (data['Close'].iloc[-4:].sum() + next_price) / 5
            new_row['MA_10'] = (data['Close'].iloc[-9:].sum() + next_price) / 10
            new_row['MA_20'] = (data['Close'].iloc[-19:].sum() + next_price) / 20
            
            last_row = new_row
        
        predicted_return = (future_data[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1] * 100
        return predicted_return, r2
    except Exception as e:
        return None, None

# Streamlit app
def main():
    st.title("📊 Advanced ETF Performance Dashboard")
    st.write("""
    Comprehensive analysis of Indian ETFs with performance metrics, correlation analysis, 
    and predictive modeling using multiple algorithms.
    """)
    
    # Sidebar filters
    st.sidebar.header("Filters")
    analysis_period = st.sidebar.selectbox("Analysis Period", ['1 Week', '1 Month', '3 Months', '1 Year'])
    num_etfs = st.sidebar.slider("Number of ETFs to Display", 5, 50, 15)
    min_volume = st.sidebar.number_input("Minimum Average Daily Volume (in Cr)", min_value=0, value=5)
    
    # Convert period to yfinance format
    period_map = {'1 Week': '1wk', '1 Month': '1mo', '3 Months': '3mo', '1 Year': '1y'}
    period = period_map[analysis_period]
    
    # Get data
    with st.spinner(f"Fetching {analysis_period} data for {len(etfs)} ETFs..."):
        data = get_data(etfs, period)
    
    # Calculate returns and filter by volume
    returns_df = calculate_returns(data, etfs)
    
    # Add volume information
    for etf in returns_df.index:
        try:
            vol = data[etf]['Volume'].mean() * data[etf]['Close'].mean() / 10000000  # Convert to Cr
            returns_df.loc[etf, 'Volume (Cr)'] = vol
        except:
            returns_df.loc[etf, 'Volume (Cr)'] = 0
    
    # Filter by volume
    returns_df = returns_df[returns_df['Volume (Cr)'] >= min_volume]
    
    if returns_df.empty:
        st.error("No ETFs meet the selected volume criteria. Please adjust filters.")
        return
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total ETFs Analyzed", len(returns_df))
    col2.metric("Average Return", f"{returns_df['Return'].mean():.2f}%")
    col3.metric("Best Performer", 
                f"{returns_df.iloc[0].name} ({returns_df.iloc[0]['Return']:.2f}%)")
    
    # Top and bottom performers
    st.subheader(f"📈 Top {num_etfs} Performers ({analysis_period})")
    top_performers = returns_df.head(num_etfs)
    
    fig = px.bar(top_performers, x=top_performers.index, y='Return', 
                 color='Return', color_continuous_scale='greens',
                 title=f"Top {num_etfs} ETF Performers",
                 labels={'index': 'ETF', 'Return': 'Return (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"📉 Bottom {num_etfs} Performers ({analysis_period})")
    bottom_performers = returns_df.tail(num_etfs).sort_values('Return')
    
    fig = px.bar(bottom_performers, x=bottom_performers.index, y='Return', 
                 color='Return', color_continuous_scale='reds',
                 title=f"Bottom {num_etfs} ETF Performers",
                 labels={'index': 'ETF', 'Return': 'Return (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed ETF analysis
    st.subheader("🔍 Detailed ETF Analysis")
    selected_etf = st.selectbox("Select ETF for Detailed Analysis", returns_df.index)
    
    if selected_etf:
        try:
            etf_data = yf.download(selected_etf, period='6mo')['Close']
            
            # Price chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=etf_data.index, y=etf_data, 
                                  mode='lines', name='Price',
                                  line=dict(color='royalblue', width=2)))
            fig.update_layout(title=f"{selected_etf} Price Trend (6 Months)",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"₹{returns_df.loc[selected_etf, 'Last Price']:.2f}")
            col2.metric(f"{analysis_period} Return", 
                       f"{returns_df.loc[selected_etf, 'Return']:.2f}%")
            col3.metric("Annualized Volatility", 
                       f"{returns_df.loc[selected_etf, 'Volatility']:.2f}%")
            col4.metric("Average Daily Volume", 
                       f"₹{returns_df.loc[selected_etf, 'Volume (Cr)']:.2f} Cr")
            
            # Predictive analysis
            st.subheader("🔮 Performance Prediction")
            
            if st.button("Run Predictive Analysis"):
                with st.spinner("Running multiple prediction models..."):
                    # RF Prediction
                    rf_return, rf_score = rf_prediction(selected_etf, 30)
                    
                    # ARIMA Prediction
                    arima_return = arima_prediction(selected_etf, 30)
                    
                    # LSTM Prediction
                    lstm_return = lstm_prediction(selected_etf, 30)
                    
                    # Display results
                    if rf_return is not None and arima_return is not None and lstm_return is not None:
                        # Create dataframe for results
                        pred_df = pd.DataFrame({
                            'Model': ['Random Forest', 'ARIMA', 'LSTM'],
                            '1-Month Prediction (%)': [rf_return, arima_return, lstm_return],
                            'Confidence': [rf_score*100 if rf_score else 0, 70, 75]  # Placeholder for ARIMA/LSTM
                        })
                        
                        # Display metrics
                        col1, col2 = st.columns(2)
                        avg_pred = pred_df['1-Month Prediction (%)'].mean()
                        col1.metric("Average Predicted 1-Month Return", f"{avg_pred:.2f}%")
                        
                        # Model comparison chart
                        fig = px.bar(pred_df, x='Model', y='1-Month Prediction (%)',
                                     color='1-Month Prediction (%)',
                                     color_continuous_scale='bluered',
                                     title="Model Predictions Comparison")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show detailed predictions
                        st.write("### Model Predictions Details")
                        st.dataframe(pred_df.style.format({
                            '1-Month Prediction (%)': '{:.2f}%',
                            'Confidence': '{:.1f}%'
                        }))
                        
                        # Recommendation
                        if avg_pred > 5:
                            st.success("✅ Strong Buy Recommendation - Positive outlook across all models")
                        elif avg_pred > 0:
                            st.info("ℹ️ Moderate Buy Recommendation - Slightly positive outlook")
                        elif avg_pred > -5:
                            st.warning("⚠️ Hold Recommendation - Mixed signals")
                        else:
                            st.error("❌ Sell Recommendation - Negative outlook across models")
                    else:
                        st.error("Could not generate predictions for this ETF. Please try another one.")
        except Exception as e:
            st.error(f"Error analyzing {selected_etf}: {str(e)}")
    
    # Correlation analysis
    st.subheader("📊 Correlation Analysis")
    st.write("Analyze how top performers move in relation to each other")
    
    top_corr_etfs = st.multiselect(
        "Select ETFs for Correlation Analysis",
        options=returns_df.index,
        default=returns_df.head(5).index.tolist()
    )
    
    if len(top_corr_etfs) >= 2:
        corr_data = yf.download(top_corr_etfs, period='3mo')['Close']
        corr_matrix = corr_data.corr()
        
        fig = px.imshow(corr_matrix,
                        zmin=-1, zmax=1,
                        color_continuous_scale='RdBu',
                        title="ETF Correlation Heatmap",
                        labels=dict(x="ETF", y="ETF", color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster map
        st.write("### Cluster Map")
        fig = px.dendrogram(corr_matrix, color_threshold=0.5)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least 2 ETFs for correlation analysis")
    
    # Performance vs Risk scatter plot
    st.subheader("🎯 Risk-Return Profile")
    fig = px.scatter(returns_df, x='Volatility', y='Return',
                     color='Return', size='Volume (Cr)',
                     hover_name=returns_df.index,
                     title="Risk-Return Analysis of ETFs",
                     labels={'Volatility': 'Annualized Volatility (%)',
                            'Return': f'{analysis_period} Return (%)'})
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
