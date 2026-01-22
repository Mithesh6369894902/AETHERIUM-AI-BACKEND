import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import timedelta


def run_forecast(payload: dict):
    symbol = payload["symbol"]
    start = payload["start_date"]
    end = payload["end_date"]
    horizon = int(payload["horizon"])

    # ---------------- LOAD DATA ---------------- #
    df = yf.download(symbol, start=start, end=end)

    if df.empty:
        return {"error": "No stock data found"}

    df = df.reset_index()
    df["TimeIndex"] = np.arange(len(df))

    # ---------------- MODEL ---------------- #
    X = df[["TimeIndex"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    # ---------------- FORECAST ---------------- #
    future_idx = np.arange(len(df), len(df) + horizon).reshape(-1, 1)
    future_prices = model.predict(future_idx)

    future_dates = [
        (df["Date"].iloc[-1] + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]

    # ---------------- METRICS ---------------- #
    recent_avg = float(y.tail(10).mean())
    future_avg = float(np.mean(future_prices))

    signal = "BUY" if future_avg > recent_avg else "SELL"
    confidence = round(abs(future_avg - recent_avg) / recent_avg, 3)

    # ---------------- RESPONSE ---------------- #
    return {
        "signal": signal,
        "confidence": confidence,
        "recent_avg": round(recent_avg, 2),
        "future_avg": round(future_avg, 2),

        # 🔥 GRAPH DATA
        "historical": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "price": round(float(p), 2)
            }
            for d, p in zip(df["Date"], df["Close"])
        ],

        "forecast": [
            {
                "date": d,
                "price": round(float(p), 2)
            }
            for d, p in zip(future_dates, future_prices)
        ]
    }

