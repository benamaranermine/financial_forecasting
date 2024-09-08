import streamlit as st
import yfinance as yf
import plotly.graph_objs as go

def get_historical_data(ticker, period='1mo'):
    # Fetch historical stock data using yfinance
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)  # Default period is 1 month
    return data

def plot_closing_price_curve(data, ticker):
    # Create a colorful line chart for closing prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines+markers',
                             line=dict(color='royalblue', width=2),
                             marker=dict(size=5, color='lightblue'),
                             name='Close Price'))
    fig.update_layout(
        title=f"<b>{ticker} Closing Price Over Time</b>",
        title_font=dict(size=24, color='darkblue', family='Arial'),
        xaxis_title='<b>Date</b>',
        yaxis_title='<b>Price (USD)</b>',
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black'),
        template='plotly_dark',
        paper_bgcolor='lightgray',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def display_homepage():
    # Set a custom background color and style for the app
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
            font-family: 'Arial';
        }
        h1 {
            color: #1f77b4;
            font-size: 42px;
            text-align: center;
        }
        .stMarkdown {
            color: #333333;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Welcome to the Stock Forecast App!')

    # Specify the stock ticker
    ticker = 'GOOGL'

    # Fetch historical data
    historical_data = get_historical_data(ticker, period='1mo')  # Last 1 month of data

    if not historical_data.empty:
        # Display the latest closing price with a custom style
        latest_close_price = historical_data['Close'].iloc[-1]
        st.markdown(f"""
        <h2 style='color: #ff5733; font-size: 28px;'>
        Latest {ticker} Closing Price: ${latest_close_price:.2f} (as of {historical_data.index[-1].date()})
        </h2>
        """, unsafe_allow_html=True)

        # Display the closing price curve
        st.plotly_chart(plot_closing_price_curve(historical_data, ticker))
    else:
        st.markdown(f"""
        <h2 style='color: #ff5733; font-size: 28px;'>
        Closing Price Information for {ticker} is not available today.
        </h2>
        """, unsafe_allow_html=True)

    # Provide a colorful description of the project
    st.markdown("""
    <div style='padding: 20px; background-color: #e8f4fa; border-radius: 10px;'>
    <h3 style='color: #2ca02c;'>About This Project</h3>
    
    Welcome to our Stock Forecast App! This application is designed to help you predict stock prices using historical data and advanced forecasting models, and to see the recommended actions (buy, sell, hold) based on the model's predictions.

    <h4 style='color: #d62728;'>Key Features:</h4>
    <ul>
    <li><b>Stock Prediction:</b> Choose from various stocks and predict their future prices based on historical data.</li>
    <li><b>Forecast Visualization:</b> View interactive plots of historical and predicted stock prices.</li>
    <li><b>Model Actions:</b> See the recommended actions (buy, sell, hold) based on the model's predictions.</li>
    </ul>

    

    <h4 style='color: #ff7f0e;'>Data and Technology:</h4>
    This project utilizes various technologies including TensorFlow, Prophet, Plotly, and MySQL to provide accurate forecasts and insightful data analysis.

    <h4 style='color: #17becf;'>Get Started</h4>
    Navigate to the options below to start exploring stock forecasts!
    </div>
    """, unsafe_allow_html=True)

# Uncomment the line below to run the app
# display_homepage()
