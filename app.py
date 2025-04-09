import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px

# Set page config
st.set_page_config(page_title="ETF Performance Dashboard", layout="wide")

# ETF list
etfs = [
    "MAFANG.NS", "FMCGIETF.NS", "MOGSEC.NS", "TATAGOLD.NS", "GOLDIETF.NS",
    "GOLDCASE.NS", "HDFCGOLD.NS", "GOLD1.NS", "AXISGOLD.NS", "GOLD360.NS",
    "ABGSEC.NS", "SETFGOLD.NS", "GOLDBEES.NS", "LICMFGOLD.NS", "QGOLDHALF.NS",
    "GSEC5IETF.NS", "IVZINGOLD.NS", "GOLDSHARE.NS", "BSLGOLDETF.NS", "LICNFNHGP.NS",
    "GOLDETFADD.NS", "UNIONGOLD.NS", "CONSUMBEES.NS", "SDL26BEES.NS", "AXISCETF.NS",
    "GROWWGOLD.NS", "GOLDETF.NS", "MASPTOP50.NS", "SETF10GILT.NS", "EBBETF0433.NS",
    "NV20BEES.NS", "BBNPPGOLD.NS", "CONSUMIETF.NS", "AUTOBEES.NS", "BSLSENETFG.NS",
    "LTGILTBEES.NS", "AUTOIETF.NS", "AXISBPSETF.NS", "GILT5YBEES.NS", "LIQUIDCASE.NS",
    "GROWWLIQID.NS", "GSEC10YEAR.NS", "LIQUIDBETF.NS", "LIQUIDADD.NS", "LIQUID1.NS",
    "HDFCLIQUID.NS", "MOLOWVOL.NS", "AONELIQUID.NS", "CASHIETF.NS", "LIQUIDPLUS.NS",
    "LIQUIDSHRI.NS", "ABSLLIQUID.NS", "LIQUIDETF.NS", "CONS.NS", "LIQUIDSBI.NS",
    "LIQUID.NS", "EGOLD.NS", "BBNPNBETF.NS", "LIQUIDIETF.NS", "IVZINNIFTY.NS",
    "GSEC10ABSL.NS", "LIQUIDBEES.NS", "EBBETF0430.NS", "SBIETFCON.NS", "MON100.NS",
    "LICNETFGSC.NS", "GSEC10IETF.NS", "QUAL30IETF.NS", "SILVRETF.NS", "LICNETFSEN.NS",
    "HDFCLOWVOL.NS", "EBANKNIFTY.NS", "LOWVOLIETF.NS", "EBBETF0431.NS", "TOP100CASE.NS",
    "NIFTYQLITY.NS", "HDFCGROWTH.NS", "SHARIABEES.NS", "BBETF0432.NS", "NETF.NS",
    "UTISENSETF.NS", "NIF10GETF.NS", "LOWVOL1.NS", "MOM50.NS", "CPSEETF.NS",
    "SBIETFPB.NS", "SILVERIETF.NS", "SENSEXETF.NS", "HDFCSENSEX.NS", "BANKBETF.NS",
    "ESILVER.NS", "HDFCNIF100.NS", "BANKETF.NS", "MNC.NS", "LOWVOL.NS", "INFRABEES.NS",
    "MID150.NS", "SENSEXADD.NS", "BANKIETF.NS", "ICICIB22.NS", "NIF5GETF.NS",
    "TOP10ADD.NS", "SILVERADD.NS", "BANKETFADD.NS", "HDFCNIFTY.NS", "SETFNIF50.NS",
    "ALPL30IETF.NS", "OILIETF.NS", "NIFTY1.NS", "GROWWEV.NS", "SILVER1.NS",
    "NIFTYBETF.NS", "LICNETFN50.NS", "SETFNIFBK.NS", "NIFTYBEES.NS", "UTIBANKETF.NS",
    "NIF100BEES.NS", "UTINEXT50.NS", "PVTBANIETF.NS", "NIFTYETF.NS", "NEXT30ADD.NS",
    "SENSEXIETF.NS", "QNIFTY.NS", "AXISILVER.NS", "NIFTYIETF.NS", "PVTBANKADD.NS",
    "SBINEQWETF.NS", "MONIFTY500.NS", "UTINIFTETF.NS", "TATSILV.NS", "NIFTY50ADD.NS",
    "BSLNIFTY.NS", "NIF100IETF.NS", "HDFCNIFBAN.NS", "AXISNIFTY.NS", "HDFCSILVER.NS",
    "SILVER.NS", "SILVERCASE.NS", "BANKBEES.NS", "IDFNIFTYET.NS", "ABSLNN50ET.NS",
    "EVINDIA.NS", "ABSLPSE.NS", "ABSLBANETF.NS", "ESG.NS", "INFRAIETF.NS",
    "SILVERBEES.NS", "SBIETFQLTY.NS", "JUNIORBEES.NS", "BANKNIFTY1.NS", "SILVER360.NS",
    "NEXT50IETF.NS", "NIFTY100EW.NS", "HDFCNEXT50.NS", "MOQUALITY.NS", "SBISILVER.NS",
    "NEXT50.NS", "CONSUMER.NS", "HEALTHIETF.NS", "GROWWN200.NS", "SETFNN50.NS",
    "MSCIINDIA.NS", "DIVOPPBEES.NS", "NV20IETF.NS", "AONETOTAL.NS", "AXISHCETF.NS",
    "SILVERETF.NS", "GROWWRAIL.NS", "EQUAL50ADD.NS", "BSE500IETF.NS", "AXSENSEX.NS",
    "AXISBNKETF.NS", "AXISVALUE.NS", "HEALTHY.NS", "EQUAL200.NS", "MAKEINDIA.NS",
    "MID150CASE.NS", "UTISXN50.NS", "NPBET.NS", "PHARMABEES.NS", "NV20.NS",
    "HDFCVALUE.NS", "BFSI.NS", "MIDCAPIETF.NS", "MULTICAP.NS", "SBIBPB.NS",
    "MIDCAP.NS", "COMMOIETF.NS", "MOVALUE.NS", "SELECTIPO.NS", "NIFMID150.NS",
    "MID150BEES.NS", "HDFCMOMENT.NS", "MODEFENCE.NS", "MIDCAPETF.NS", "MOM100.NS",
    "HDFCMID150.NS", "MOSMALL250.NS", "MOMENTUM.NS", "MIDSELIETF.NS", "HDFCSML250.NS",
    "PSUBANKADD.NS", "MOCAPITAL.NS", "LICNMID100.NS", "MOM30IETF.NS", "FINIETF.NS",
    "MOHEALTH.NS", "SMALLCAP.NS", "PSUBANK.NS", "HEALTHADD.NS", "MIDQ50ADD.NS",
    "MOMOMENTUM.NS", "HDFCBSE500.NS", "GROWWDEFNC.NS", "ALPHAETF.NS", "EMULTIMQ.NS",
    "MIDSMALL.NS", "PSUBNKIETF.NS", "PSUBNKBEES.NS", "HDFCPSUBK.NS", "MOMENTUM50.NS",
    "ALPHA.NS", "METALIETF.NS", "BANKPSU.NS", "TNIDETF.NS", "ECAPINSURE.NS",
    "HNGSNGBEES.NS", "MOREALTY.NS", "METAL.NS", "VAL30IETF.NS", "HDFCNIFIT.NS",
    "TECH.NS", "ITIETF.NS", "ITETF.NS", "SBIETFIT.NS", "AXISTECETF.NS", "NIFITETF.NS",
    "ITBEES.NS", "MAHKTECH.NS", "IT.NS", "ITETFADD.NS", "HDFCPVTBAN.NS",
    "HDFCQUAL.NS", "MONQ50.NS"
]

# Function to fetch data
@st.cache_data
def get_data(etf_list, period='1mo'):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30) if period == '1mo' else end_date - timedelta(days=7)
    
    data = yf.download(etf_list, start=start_date, end=end_date, group_by='ticker')
    return data

# Function to calculate returns
def calculate_returns(data, etf_list):
    returns = {}
    for etf in etf_list:
        try:
            close_prices = data[etf]['Close']
            if len(close_prices) > 1:
                returns[etf] = (close_prices[-1] - close_prices[0]) / close_prices[0] * 100
        except:
            continue
    return pd.DataFrame.from_dict(returns, orient='index', columns=['Return']).sort_values('Return', ascending=False)

# Function for predictive analysis
def predict_future_performance(etf, days=30):
    try:
        # Get historical data
        data = yf.download(etf, period='1y')['Close'].reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        data['Day'] = (data['Date'] - data['Date'].min()).dt.days
        
        # Prepare features
        X = data[['Day']]
        y = data['Close']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        future_days = pd.DataFrame({'Day': range(data['Day'].max()+1, data['Day'].max()+days+1)})
        future_prices = model.predict(future_days)
        
        # Calculate predicted return
        current_price = data['Close'].iloc[-1]
        predicted_price = future_prices[-1]
        predicted_return = (predicted_price - current_price) / current_price * 100
        
        return predicted_return, model.score(X_test, y_test)
    except:
        return None, None

# Streamlit app
def main():
    st.title("ETF Performance Dashboard")
    st.write("Analyzing performance of Indian ETFs using yfinance data")
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        analysis_period = st.selectbox("Select Analysis Period", ['1 Week', '1 Month'])
    with col2:
        num_etfs = st.slider("Number of ETFs to display", 5, 50, 10)
    
    # Get data
    period = '1wk' if analysis_period == '1 Week' else '1mo'
    data = get_data(etfs, period)
    
    # Calculate returns
    returns_df = calculate_returns(data, etfs)
    
    # Display top and bottom performers
    st.subheader(f"Top {num_etfs} Performers ({analysis_period})")
    top_performers = returns_df.head(num_etfs)
    st.dataframe(top_performers.style.format({'Return': '{:.2f}%'}))
    
    # Plot top performers
    fig = px.bar(top_performers, x=top_performers.index, y='Return', 
                 title=f"Top {num_etfs} ETF Performers ({analysis_period})",
                 labels={'index': 'ETF', 'Return': 'Return (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader(f"Bottom {num_etfs} Performers ({analysis_period})")
    bottom_performers = returns_df.tail(num_etfs).sort_values('Return')
    st.dataframe(bottom_performers.style.format({'Return': '{:.2f}%'}))
    
    # Plot bottom performers
    fig = px.bar(bottom_performers, x=bottom_performers.index, y='Return', 
                 title=f"Bottom {num_etfs} ETF Performers ({analysis_period})",
                 labels={'index': 'ETF', 'Return': 'Return (%)'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive analysis section
    st.subheader("Predictive Analysis for Next Period")
    
    # Select ETF for prediction
    selected_etf = st.selectbox("Select ETF for Predictive Analysis", returns_df.index)
    
    if st.button("Predict Performance"):
        with st.spinner("Running predictive analysis..."):
            predicted_return_1w, score_1w = predict_future_performance(selected_etf, 7)
            predicted_return_1m, score_1m = predict_future_performance(selected_etf, 30)
            
            if predicted_return_1w is not None and predicted_return_1m is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted 1-Week Return", f"{predicted_return_1w:.2f}%", 
                             delta_color="inverse" if predicted_return_1w < 0 else "normal")
                    st.write(f"Model R² Score: {score_1w:.2f}")
                
                with col2:
                    st.metric("Predicted 1-Month Return", f"{predicted_return_1m:.2f}%", 
                             delta_color="inverse" if predicted_return_1m < 0 else "normal")
                    st.write(f"Model R² Score: {score_1m:.2f}")
                
                # Get historical data for chart
                hist_data = yf.download(selected_etf, period='6mo')['Close']
                fig = px.line(hist_data, title=f"{selected_etf} Historical Performance (6 Months)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not perform prediction for this ETF. Please try another one.")
    
    # Correlation analysis
    st.subheader("ETF Correlation Analysis")
    st.write("Analyzing correlation between top performers")
    
    # Get data for correlation
    top_etfs = returns_df.head(10).index.tolist()
    correlation_data = yf.download(top_etfs, period='1mo')['Close']
    
    if len(top_etfs) > 1:
        correlation_matrix = correlation_data.corr()
        
        fig = px.imshow(correlation_matrix,
                        labels=dict(x="ETF", y="ETF", color="Correlation"),
                        x=correlation_matrix.columns,
                        y=correlation_matrix.columns,
                        title="Correlation Matrix of Top Performers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for correlation analysis")

if __name__ == "__main__":
    main()
