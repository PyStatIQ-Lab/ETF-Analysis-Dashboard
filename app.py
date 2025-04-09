import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_dendrogram

# Set page config
st.set_page_config(page_title="📊 ETF Comparison Dashboard", layout="wide")

# ETF list (optimized for reliability)
etfs = [
    "GOLDBEES.NS", "NIFTYBEES.NS", "BANKBEES.NS", "JUNIORBEES.NS", 
    "LIQUIDBEES.NS", "NIF100BEES.NS", "NV20BEES.NS", "SETFNIF50.NS", 
    "HDFCNIFTY.NS", "ICICINIFTY.NS", "AXISNIFTY.NS", "SBIETFQLTY.NS",
    "UTINIFTETF.NS", "SILVERBEES.NS", "ITBEES.NS", "CPSEETF.NS",
    "INFRABEES.NS", "PHARMABEES.NS", "AUTOBEES.NS", "HDFCGOLD.NS",
    "HDFCSILVER.NS", "HDFCNIF100.NS", "HDFCSENSEX.NS", "HDFCNEXT50.NS"
]

@st.cache_data(ttl=3600)
def get_etf_data(etf_list, period='1y'):
    end_date = datetime.today()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
    elif period == '1mo':
        start_date = end_date - timedelta(days=30)
    elif period == '3mo':
        start_date = end_date - timedelta(days=90)
    else:  # 1 year default
        start_date = end_date - timedelta(days=365)
    
    return yf.download(etf_list, start=start_date, end=end_date, group_by='ticker', progress=False)

def calculate_metrics(data, etf_list):
    metrics = {}
    for etf in etf_list:
        try:
            if etf in data:
                close = data[etf]['Close']
                returns = (close[-1] - close[0]) / close[0] * 100
                volatility = np.std(close.pct_change().dropna()) * np.sqrt(252) * 100
                metrics[etf] = {
                    'Return%': returns,
                    'Volatility%': volatility,
                    'Last Price': close[-1],
                    'Volume (Cr)': data[etf]['Volume'].mean() * close.mean() / 1e7
                }
        except:
            continue
    return pd.DataFrame.from_dict(metrics, orient='index')

# Main App
def main():
    st.title("📊 ETF Comparison Dashboard")
    st.write("Compare ETFs to identify the best investment opportunities")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    period = st.sidebar.selectbox("Analysis Period", ['1 Week', '1 Month', '3 Months', '1 Year'])
    period_map = {'1 Week': '1wk', '1 Month': '1mo', '3 Months': '3mo', '1 Year': '1y'}
    
    # Get data
    with st.spinner("Loading ETF data..."):
        data = get_etf_data(etfs, period_map[period])
        metrics_df = calculate_metrics(data, etfs)
    
    # Main comparison section
    st.header("🔍 ETF Comparison Tool")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_etfs = st.multiselect(
            "Select ETFs to compare (2-5 recommended)", 
            options=metrics_df.index.tolist(),
            default=metrics_df.nlargest(3, 'Return%').index.tolist()
        )
    
    with col2:
        compare_by = st.selectbox(
            "Comparison metric",
            options=['Return%', 'Volatility%', 'Last Price', 'Volume (Cr)'],
            index=0
        )
    
    if len(selected_etfs) >= 1:
        # Metric comparison chart
        fig = px.bar(
            metrics_df.loc[selected_etfs].reset_index(), 
            x='index', y=compare_by,
            color=compare_by,
            color_continuous_scale='Viridis',
            title=f"ETF Comparison by {compare_by}",
            labels={'index': 'ETF', compare_by: compare_by}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series comparison
        if len(selected_etfs) > 1:
            st.subheader("📈 Price Trend Comparison")
            
            # Normalized price comparison
            norm_prices = pd.DataFrame()
            for etf in selected_etfs:
                try:
                    prices = data[etf]['Close']
                    norm_prices[etf] = (prices / prices.iloc[0]) * 100  # Normalize to 100
                except:
                    continue
            
            if not norm_prices.empty:
                fig = px.line(
                    norm_prices, 
                    title="Normalized Price Comparison (Base=100)",
                    labels={'value': 'Normalized Price', 'variable': 'ETF'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation analysis
                st.subheader("📊 Correlation Analysis")
                corr_matrix = norm_prices.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    zmin=-1, zmax=1,
                    color_continuous_scale='RdBu',
                    title="ETF Price Correlation",
                    labels=dict(color="Correlation")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk-Return scatter plot
                st.subheader("🎯 Risk-Return Profile")
                scatter_df = metrics_df.loc[selected_etfs].reset_index()
                
                fig = px.scatter(
                    scatter_df,
                    x='Volatility%', 
                    y='Return%',
                    color='Return%',
                    size='Volume (Cr)',
                    hover_name='index',
                    title="Risk vs Return Comparison",
                    labels={
                        'Volatility%': 'Annualized Volatility (%)',
                        'Return%': f'{period} Return (%)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Performance tables
    st.header("📋 Performance Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Performers")
        st.dataframe(
            metrics_df.nlargest(10, 'Return%').style.format({
                'Return%': '{:.2f}%',
                'Volatility%': '{:.2f}%',
                'Last Price': '₹{:.2f}',
                'Volume (Cr)': '₹{:.2f} Cr'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Lowest Volatility")
        st.dataframe(
            metrics_df.nsmallest(10, 'Volatility%').style.format({
                'Return%': '{:.2f}%',
                'Volatility%': '{:.2f}%',
                'Last Price': '₹{:.2f}',
                'Volume (Cr)': '₹{:.2f} Cr'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
