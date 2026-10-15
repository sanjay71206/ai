---
title: StockVision AI
emoji: 📈
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# StockVision AI

FastAPI + TensorFlow/Keras LSTM + yfinance + A* strategy planner, with the glassmorphism UI served at `/`.
Educational only. Forecasts are model estimates, not advice or guarantees.

## Run (Python 3.10–3.12)
```bash
cd stockvision
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env               # optional
uvicorn backend.main:app --reload --port 8000
```
Open http://localhost:8000 — the header badge shows **● LIVE · yfinance via FastAPI**.
If the backend is unreachable the UI falls back to a clearly labelled synthetic DEMO mode.

## Test
```bash
python -m pytest backend/tests -v
```
(LSTM test is skipped automatically if TensorFlow is missing.)

## API
GET `/api/health` · GET `/api/stock/{ticker}?period=5y` · POST `/api/model/train` `{ticker,lookback,epochs}` ·
POST `/api/predict` `{ticker,horizon}` · GET `/api/model/metrics/{ticker}` · POST `/api/strategy/astar` `{scenario,start_equity}`
Interactive docs: http://localhost:8000/docs

## Notes
- Chronological 70/15/15 split; MinMaxScaler fit on train only; early stopping on the validation set; naive previous-close baseline.
- Trained models/metrics are saved in `backend/ml/saved/`.
- Indian tickers need a suffix (e.g. `RELIANCE.NS`). Forecast dates skip weekends only, not exchange holidays.
- Training runs on the server and can take 10–60 s on CPU.
