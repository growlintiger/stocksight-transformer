import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.preprocessing import StandardScaler
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

def create_sequences(values, window_size):
    X, y = [], []
    for i in range(len(values) - window_size):
        seq = values[i:i + window_size]
        weights = np.linspace(0.5, 1.5, window_size).reshape(-1, 1)
        X.append(seq * weights)
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

tickers = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS",
    "SBIN.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "HCLTECH.NS", "LT.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "AXISBANK.NS", "SUNPHARMA.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "WIPRO.NS", "POWERGRID.NS",
    "TITAN.NS", "ONGC.NS", "NTPC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "NESTLEIND.NS", "TATASTEEL.NS",
    "HDFCLIFE.NS", "TECHM.NS", "BAJAJFINSV.NS", "COALINDIA.NS", "JSWSTEEL.NS", "GRASIM.NS",
    "BRITANNIA.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "DRREDDY.NS", "TATAMOTORS.NS",
    "DIVISLAB.NS", "CIPLA.NS", "HINDALCO.NS", "BPCL.NS", "UPL.NS", "INDUSINDBK.NS", "SHREECEM.NS",
    "BAJAJ-AUTO.NS", "HDFCAMC.NS", "APOLLOHOSP.NS", "ICICIPRULI.NS", "M&M.NS"
]

data_dir = "data"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

window_size = 60
epochs = 50
batch_size = 64

print(f"Starting training for {len(tickers)} stocks...\n")

for idx, ticker in enumerate(tickers, 1):
    try:
        csv_path = os.path.join(data_dir, f"{ticker}.csv")
        model_path = os.path.join(model_dir, f"{ticker}_returns_transformer.h5")

        if not os.path.exists(csv_path):
            print(f"[{idx}/{len(tickers)}] Missing data for {ticker}, skipping.")
            continue
        if os.path.exists(model_path):
            print(f"[{idx}/{len(tickers)}] Model already exists for {ticker}, skipping.")
            continue

        data = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        df = data[["Close"]].dropna()
        df["Return"] = df["Close"].pct_change()
        df = df.dropna()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[["Return"]])
        X, y = create_sequences(scaled, window_size)
        if len(X) < 20:
            print(f"[{idx}/{len(tickers)}] Not enough data for {ticker}, skipping.")
            continue

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        X_train, X_test = X_train.reshape((-1, window_size, 1)), X_test.reshape((-1, window_size, 1))

        model = build_time_transformer(window_size)
        es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

        model.save(model_path)
        print(f"[{idx}/{len(tickers)}] Trained and saved model for {ticker}")

    except Exception as e:
        print(f"[{idx}/{len(tickers)}] Error training {ticker}: {e}")

print("\nAll stocks processed. Models saved in /models folder.")
