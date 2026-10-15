"""Historical data via yfinance: validation, cleaning, short-lived cache."""
import re
import time

import pandas as pd

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,12}$")
PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
_CACHE: dict = {}
TTL = 300  # seconds


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


def validate_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not TICKER_RE.match(t):
        raise ApiError(422, f"Invalid ticker symbol: {t!r}")
    return t


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Close/Volume, drop duplicates, forward-fill gaps, drop unusable rows."""
    df = df[["Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").where(lambda s: s > 0).ffill()
    df = df.dropna(subset=["Close"])
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    return df


def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    ticker = validate_ticker(ticker)
    if period not in PERIODS:
        raise ApiError(422, f"Invalid period {period!r}. Use one of {sorted(PERIODS)}")
    hit = _CACHE.get((ticker, period))
    if hit and time.time() - hit[0] < TTL:
        return hit[1].copy()
    try:
        import yfinance as yf
    except ImportError:
        raise ApiError(503, "yfinance is not installed. Run: pip install -r backend/requirements.txt")
    try:
        raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception as e:  # network, rate limit, parsing
        msg = str(e)
        if "429" in msg or "Too Many" in msg or "RateLimit" in type(e).__name__:
            raise ApiError(429, "Rate limited by Yahoo Finance. Wait a minute and retry.")
        raise ApiError(502, f"Could not reach Yahoo Finance: {msg[:150]}")
    if raw is None or raw.empty:
        raise ApiError(404, f"No data found for '{ticker}'. Check the symbol (Indian stocks need a suffix, e.g. RELIANCE.NS).")
    df = clean_frame(raw)
    if len(df) < 2:
        raise ApiError(404, f"Not enough usable data for '{ticker}'.")
    _CACHE[(ticker, period)] = (time.time(), df)
    return df.copy()
