from fastapi import APIRouter

from backend.services.data_service import fetch_history, validate_ticker

router = APIRouter()


@router.get("/api/stock/{ticker}")
def get_stock(ticker: str, period: str = "5y"):
    t = validate_ticker(ticker)
    df = fetch_history(t, period)
    return {"ticker": t, "source": "yfinance", "period": period, "count": len(df),
            "last_date": df.index[-1].strftime("%Y-%m-%d"),
            "dates": [d.strftime("%Y-%m-%d") for d in df.index],
            "close": [round(float(x), 4) for x in df["Close"]],
            "volume": [int(v) for v in df["Volume"]]}
