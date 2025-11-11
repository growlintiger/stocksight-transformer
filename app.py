import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import plotly.graph_objects as go
import os
import plotly.io as pio

pio.templates.default = "plotly_white"

def create_sequences(values, window_size):
    X, y = [], []
    for i in range(len(values) - window_size):
        seq = values[i:i + window_size]
        weights = np.linspace(0.5, 1.5, window_size).reshape(-1, 1)
        seq_weighted = seq * weights
        X.append(seq_weighted)
        y.append(values[i + window_size])
    return np.array(X), np.array(y)

def get_positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)

def build_time_transformer(input_len, d_model=128, num_heads=8, ff_dim=256, num_layers=3, dropout_rate=0.1):
    inputs = layers.Input(shape=(input_len, 1))
    x = layers.Dense(d_model)(inputs)
    pos_enc = get_positional_encoding(input_len, d_model)
    x = x + pos_enc
    for _ in range(num_layers):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_out = layers.Dropout(dropout_rate)(attn_out)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        ff = layers.Dropout(dropout_rate)(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(1e-4), loss=losses.MeanSquaredError())
    return model

st.set_page_config(page_title="StockSight - GTT Model", page_icon="logo.jpg", layout="wide")
st.title("StockSight — Return-based Gated Time Transformer (1-Month Forecast)")
st.caption("Developed as part of the GTT Project by Group 2, KIIT University")

st.sidebar.header("Stock Selection")
tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
    "HINDUNILVR.NS", "KOTAKBANK.NS", "HCLTECH.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS",
    "SUNPHARMA.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "WIPRO.NS", "POWERGRID.NS", "TITAN.NS", "ONGC.NS",
    "NTPC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "NESTLEIND.NS", "TATASTEEL.NS", "HDFCLIFE.NS", "TECHM.NS",
    "BAJAJFINSV.NS", "COALINDIA.NS", "JSWSTEEL.NS", "GRASIM.NS", "BRITANNIA.NS", "SBILIFE.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "DRREDDY.NS", "TATAMOTORS.NS", "DIVISLAB.NS", "CIPLA.NS", "HINDALCO.NS", "BPCL.NS",
    "UPL.NS", "INDUSINDBK.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS", "HDFCAMC.NS", "APOLLOHOSP.NS", "ICICIPRULI.NS",
    "M&M.NS"
]
ticker = st.sidebar.selectbox("Choose Stock", tickers)
train_button = st.sidebar.button("Train / Retrain Model")

window_size = 60
epochs = 50
batch_size = 64

csv_path = f"data/{ticker}.csv"
if not os.path.exists(csv_path):
    st.error(f"Missing data file: {csv_path}")
    st.stop()

data = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
df = data[["Close"]].dropna()

st.write(f"Data range: {df.index.min().date()} → {df.index.max().date()} ({len(df)} rows)")

fig_raw = go.Figure()
fig_raw.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close Price'))
fig_raw.update_layout(title=f"{ticker} — 3-Year Historical Close Price", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig_raw, width='stretch')

df["Return"] = df["Close"].pct_change()
df = df.dropna()

scaler = StandardScaler()
scaled = scaler.fit_transform(df[["Return"]])

if len(df) < window_size + 30:
    window_size = max(10, len(df) // 4)

X, y = create_sequences(scaled, window_size)
if len(X) == 0 or len(y) == 0:
    st.error("Not enough data to train model.")
    st.stop()

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train, X_test = X_train.reshape((-1, window_size, 1)), X_test.reshape((-1, window_size, 1))

model_path = f"models/{ticker}_returns_transformer.h5"
os.makedirs("models", exist_ok=True)

if train_button:
    with st.spinner(f"Training GTT model for {ticker}..."):
        try:
            tf.keras.backend.clear_session()
            model = build_time_transformer(window_size)
            es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test),
                      epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
            model.save(model_path)
            st.success(f"Model trained and saved for {ticker}")
        except Exception as e:
            st.error(f"Training failed: {e}")
else:
    if os.path.exists(model_path):
        try:
            model = build_time_transformer(window_size)
            model.load_weights(model_path)
            st.info(f"Loaded saved model for {ticker}")
        except Exception as e:
            st.warning(f"Could not load model. Please retrain. Error: {e}")
    else:
        st.warning("No saved model found. Please train the model first.")

if len(X_test) > 0:
    preds_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))
    error_percent = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    st.metric("Model Error (Percentage)", f"{error_percent:.2f}%")

st.subheader("Next 30-Day Forecast")
try:
    last_window = scaled[-window_size:].reshape(1, window_size, 1)
    future_returns = []

    for _ in range(30):
        next_pred = model.predict(last_window, verbose=0)
        last_window = np.concatenate((last_window[:, 1:, :], next_pred.reshape(1, 1, 1)), axis=1)
        future_returns.append(next_pred[0, 0])

    future_returns = scaler.inverse_transform(np.array(future_returns).reshape(-1, 1)).flatten()
    last_close = df["Close"].iloc[-1]
    future_prices = [last_close]
    for r in future_returns:
        future_prices.append(future_prices[-1] * (1 + r))
    future_prices = np.array(future_prices[1:])
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices})

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=df.index[-180:], y=df["Close"].tail(180), mode='lines', name='Recent Actual (6M)'))
    fig_future.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Predicted Price"],
                                    mode='lines+markers', name='Forecast (Next 30D)'))
    fig_future.update_layout(title=f"{ticker} — 1-Month Forecast (Time vs Price)",
                             xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_future, width='stretch')

    st.dataframe(forecast_df, use_container_width=True)
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast as CSV", csv, f"{ticker}_forecast.csv", "text/csv")
except Exception as e:
    st.error(f"Forecasting failed: {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: right; color: gray; font-size: 0.85rem;'>
<b>GTT Prediction Project by Group 2</b><br>
Devyanshu Bharti (2229029)<br>
Saumyajit Chatterjee (2229086)<br>
Yash Srivastava (2229082)<br>
Abhinav Baranwal (2229085)<br>
Chirayil Alex Binu (2229110)
</div>
""", unsafe_allow_html=True)
