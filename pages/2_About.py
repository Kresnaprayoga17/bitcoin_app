import streamlit as st

st.set_page_config(layout="wide", page_title="About")

st.title("About This Application")

st.markdown("""
## Bitcoin Price Prediction Application

This application is designed to provide real-time Bitcoin price analysis and predictions using advanced machine learning techniques.

### Features
- Real-time Bitcoin price monitoring
- Interactive price charts and analysis
- LSTM-based price predictions
- Historical price analysis
- Multiple prediction timeframes

### Technology Stack
- **Python**: Primary programming language
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning framework for LSTM model
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **yfinance**: Real-time financial data

### Developer Information
- **Name**: Kresna Prayoga
- **NIM**: 211351071
- **Institution**: STT Wastukancana Purwakarta

### How It Works
The application uses Long Short-Term Memory (LSTM) neural networks to analyze historical Bitcoin price patterns and make predictions. LSTM is particularly well-suited for this task because it can:
- Remember long-term patterns
- Identify trends in time series data
- Handle the complexity of cryptocurrency price movements

### Contact
For any questions or feedback about this application, please contact:
- Email: kresnaprayogaacckuliah@gmail.com
- GitHub: github.com/kresnaprayoga17

### Disclaimer
This application is for educational purposes only. Cryptocurrency trading involves substantial risk, and past performance does not guarantee future results.
""")
