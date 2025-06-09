import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def add_range_selector(fig):
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(count=1, label='1y', step='year', stepmode='backward'),
                    dict(step='all')
                ]
            )
        ),
        xaxis_type='date'
    )

def clean_column_names(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        return data.droplevel(level=1, axis=1)
    if any(isinstance(col, tuple) for col in data.columns):
        return data.rename(columns=lambda x: x[0] if isinstance(x, tuple) else x)
    if any(ticker in str(col) for col in data.columns):
        return data.rename(columns=lambda x: str(x).replace(f"{ticker} ", "").replace(f"{ticker}_", ""))
    return data

def main():
    st.set_page_config(layout="wide", page_title="Bitcoin Dashboard")

    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Using default styles.")

    st.title('BITCOIN PRICE DASHBOARD')
    st.markdown("""
    Welcome to the Bitcoin Price Dashboard! This application provides real-time Bitcoin price analysis 
    and predictions using LSTM (Long Short-Term Memory) neural networks.
    """)

    ticker = "BTC-USD"
    try:
        data = yf.download(tickers=ticker, start='2021-01-01', progress=False)
        data = clean_column_names(data, ticker)
        if data.empty:
            st.error("Failed to fetch data. Please check your internet connection.")
            return
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return

    # Key Metrics Section
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            current_price = data['Close'].iloc[-1]
            price_change = current_price - data['Close'].iloc[-2]
            st.metric("Current Price", f"${current_price:,.2f}", 
                     delta=f"${price_change:,.2f}")
        except Exception as e:
            st.error(f"Error calculating current price: {str(e)}")

    with col2:
        try:
            daily_high = data['High'].iloc[-1]
            high_change = daily_high - data['High'].iloc[-2]
            st.metric("Daily High", f"${daily_high:,.2f}", 
                     delta=f"${high_change:,.2f}")
        except Exception as e:
            st.error(f"Error calculating daily high: {str(e)}")

    with col3:
        try:
            daily_volume = data['Volume'].iloc[-1]
            volume_change = daily_volume - data['Volume'].iloc[-2]
            st.metric("24h Volume", f"${daily_volume:,.0f}", 
                     delta=f"${volume_change:,.0f}")
        except Exception as e:
            st.error(f"Error calculating volume: {str(e)}")

    # Price Chart
    st.subheader("Price Analysis")
    chart_col1, chart_col2 = st.columns([3, 1])

    with chart_col1:
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            add_range_selector(fig)
            fig.update_layout(
                title=f"{ticker} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                height=500,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating price chart: {str(e)}")

    with chart_col2:
        try:
            st.write("Recent Price Data")
            st.dataframe(
                data[['Open', 'High', 'Low', 'Close', 'Volume']]
                .tail()
                .style.format({
                    'Open': '${:,.2f}',
                    'High': '${:,.2f}',
                    'Low': '${:,.2f}',
                    'Close': '${:,.2f}',
                    'Volume': '{:,.0f}'
                })
            )
        except Exception as e:
            st.error(f"Error displaying recent data: {str(e)}")

    # Price Trends
    st.subheader("Price Trends")
    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        try:
            fig = px.line(data.tail(30), y='Close', title='30-Day Price Trend')
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating price trend chart: {str(e)}")

    with trend_col2:
        try:
            fig = px.line(data.tail(30), y='Volume', title='30-Day Volume Trend')
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating volume trend chart: {str(e)}")

    st.markdown("""
    ### Navigation
    - Use the sidebar to navigate between different pages
    - Go to the **Prediction** page to see Bitcoin price predictions
    - Visit the **About** page to learn more about this application
    """)

if __name__ == '__main__':
    main()
