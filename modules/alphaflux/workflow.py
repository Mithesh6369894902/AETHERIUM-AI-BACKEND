import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

def run_forecast(payload: dict):
    """
    AlphaFlux forecasting engine.
    Input:
        {
            "symbol": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "horizon": 10
        }
    Output:
        {
            "signal": "BUY | SELL | HOLD",
            "confidence": float,
            "recent_avg": float,
            "future_avg": float
        }
    """

    symbol = payload["symbol"]
    start = payload["start_date"]
    end = payload["end_date"]
    horizon = int(payload.get("horizon", 10))

    # Download stock data
    data = yf.download(symbol, start=start, end=end)

    if data.empty:
        return {
            "signal": "NO DATA",
            "confidence": 0.0,
            "recent_avg": 0.0,
            "future_avg": 0.0
        }

    # Time index
    data["t"] = np.arange(len(data))
    X = data[["t"]]
    y = data["Close"]

    # Train simple trend model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast future
    future_t = np.arange(len(data), len(data) + horizon).reshape(-1, 1)
    future_prices = model.predict(future_t)

    recent_avg = float(np.mean(y.tail(10)))
    future_avg = float(np.mean(future_prices))

    delta = future_avg - recent_avg
    confidence = abs(delta) / (recent_avg + 1e-6)

    if delta > 0:
        signal = "BUY"
    elif delta < 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "confidence": round(confidence, 3),
        "recent_avg": round(recent_avg, 2),
        "future_avg": round(future_avg, 2)
    }

