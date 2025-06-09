import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

def prepare_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x_all = []
    y_all = []
    
    for i in range(lookback, len(scaled_data)):
        x_all.append(scaled_data[i-lookback:i, 0])
        y_all.append(scaled_data[i, 0])
    
    x_all = np.array(x_all)
    y_all = np.array(y_all)
    
    x_all = np.reshape(x_all, (x_all.shape[0], x_all.shape[1], 1))
    
    return x_all, y_all, scaler

def predict_future(model, last_sequence, scaler, n_steps):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        current_sequence_reshaped = current_sequence.reshape((1, current_sequence.shape[0], 1))
        next_pred = model.predict(current_sequence_reshaped, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
        
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)

st.set_page_config(layout="wide", page_title="Bitcoin Price Prediction")

st.title('Bitcoin Price Prediction')
st.markdown("""
This page uses a pre-trained LSTM (Long Short-Term Memory) neural network to predict Bitcoin prices.
The model has been trained on historical data to identify patterns and predict future price movements.
""")

ticker = "BTC-USD"

try:
    data = yf.download(tickers=ticker, start='2021-01-01', progress=False)

    if data.empty:
        st.error("Failed to fetch data. Please check your internet connection.")
    else:
        # Prepare data for LSTM
        x_all, y_all, scaler = prepare_data(data['Close'])

        # Split data
        train_size = int(len(x_all) * 0.95)
        x_train, y_train = x_all[:train_size], y_all[:train_size]
        x_test, y_test = x_all[train_size:], y_all[train_size:]

        # Select prediction days
        prediction_days = st.selectbox(
            'Select prediction period',
            options=[1, 3, 7],
            format_func=lambda x: f'{x} days'
        )

        if st.button('Generate Predictions'):
            with st.spinner(f'Training model and generating {prediction_days}-day prediction...'):
                # Create and train model from scratch
                from keras.models import Sequential
                from keras.layers import LSTM, Dense

                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='nadam', loss='mean_squared_error')

                model.fit(x_train, y_train, batch_size=1, epochs=1, verbose=0)

                # Make predictions
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                actual = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'Date': data.index[train_size+60:],
                    'Actual': actual.flatten(),
                    'Predicted': predictions.flatten()
                }).set_index('Date')

                # Display model performance
                st.subheader('Model Performance')
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
                    st.metric("RMSE", f"${rmse:,.2f}")

                with metrics_col2:
                    mae = np.mean(np.abs(predictions - actual))
                    st.metric("MAE", f"${mae:,.2f}")

                with metrics_col3:
                    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
                    st.metric("MAPE", f"{mape:.2f}%")

                # Plot predictions vs actual
                st.subheader('Actual vs Predicted Prices')
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Actual'],
                                        name='Actual', line=dict(color='blue'), hoverinfo='none'))
                fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted'],
                                        name='Predicted', line=dict(color='red'), hoverinfo='none'))

                fig.update_layout(
                    title='LSTM Model: Actual vs Predicted Bitcoin Prices',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=500,
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Future predictions section removed as per request

                # Display prediction table only
                st.write(f"Predicted prices for next {prediction_days} days:")
                st.dataframe(
                    pred_df.style.format({'Actual': '${:,.2f}', 'Predicted': '${:,.2f}'})
                )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
