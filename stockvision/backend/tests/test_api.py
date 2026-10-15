import heapq

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.algorithms.astar import MOVES, SCENARIOS, astar, transition_cost
from backend.main import app
from backend.services.data_service import ApiError, clean_frame

c = TestClient(app)


def fake_df(n=400, seed=0):
    r = np.random.default_rng(seed)
    p = 100 * np.exp(np.cumsum(r.normal(3e-4, 0.01, n)))
    return pd.DataFrame({"Close": p, "Volume": np.full(n, 1_000_000)}, index=pd.bdate_range("2023-01-02", periods=n))


def test_health():
    assert c.get("/api/health").json()["status"] == "ok"


def test_invalid_ticker():
    assert c.get("/api/stock/BAD!!").status_code == 422


def test_stock_ok(monkeypatch):
    monkeypatch.setattr("backend.routes.stock.fetch_history", lambda t, p: fake_df())
    j = c.get("/api/stock/aapl?period=1y").json()
    assert j["ticker"] == "AAPL" and len(j["dates"]) == len(j["close"]) == 400


def test_no_data(monkeypatch):
    def boom(t, p):
        raise ApiError(404, "No data found for 'ZZZZ'")
    monkeypatch.setattr("backend.routes.stock.fetch_history", boom)
    r = c.get("/api/stock/ZZZZ")
    assert r.status_code == 404 and "No data" in r.json()["detail"]


def test_clean_frame_fills_and_dedupes():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"])
    df = pd.DataFrame({"Close": [10, 11, 11, np.nan], "Volume": [1, 2, 2, np.nan]}, index=idx)
    out = clean_frame(df)
    assert len(out) == 3 and out["Close"].iloc[-1] == 11 and out["Volume"].iloc[-1] == 0


def test_predict_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.ml.lstm_model.MODEL_DIR", str(tmp_path))
    assert c.post("/api/predict", json={"ticker": "AAPL", "horizon": 5}).status_code == 404


def test_request_validation():
    assert c.post("/api/predict", json={"ticker": "AAPL", "horizon": 999}).status_code == 422
    assert c.post("/api/model/train", json={"ticker": "A B", "lookback": 20}).status_code == 422


@pytest.mark.parametrize("k", list(SCENARIOS))
def test_astar_path_valid(k):
    r = astar(k)
    p, risk = r["path"], SCENARIOS[k]["risk"]
    assert p[0] == 50 and p[-1] == SCENARIOS[k]["target"]
    assert all(abs(b - a) in (10, 20) for a, b in zip(p, p[1:]))
    assert abs(r["cost"] - sum(transition_cost(a, b, risk) for a, b in zip(p, p[1:]))) < 1e-9


@pytest.mark.parametrize("k", list(SCENARIOS))
def test_astar_is_optimal(k):  # compare against Dijkstra
    risk, tgt = SCENARIOS[k]["risk"], SCENARIOS[k]["target"]
    dist, pq = {50: 0.0}, [(0.0, 50)]
    while pq:
        d, e = heapq.heappop(pq)
        if d > dist[e]:
            continue
        for m in MOVES:
            n = e + m
            if 0 <= n <= 100 and d + transition_cost(e, n, risk) < dist.get(n, 1e9):
                dist[n] = d + transition_cost(e, n, risk)
                heapq.heappush(pq, (dist[n], n))
    assert abs(astar(k)["cost"] - dist[tgt]) < 1e-9


def test_astar_endpoint():
    j = c.post("/api/strategy/astar", json={"scenario": "g", "start_equity": 50}).json()
    assert j["path"][-1] == 90 and "disclaimer" in j
    assert c.post("/api/strategy/astar", json={"scenario": "x"}).status_code == 422
    assert c.post("/api/strategy/astar", json={"scenario": "b", "start_equity": 55}).status_code == 422


def test_lstm_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    monkeypatch.setattr("backend.ml.lstm_model.MODEL_DIR", str(tmp_path))
    df = fake_df(400)
    monkeypatch.setattr("backend.routes.model.fetch_history", lambda t, p: df)
    r = c.post("/api/model/train", json={"ticker": "TEST", "lookback": 10, "epochs": 2})
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["test_dates"][0] == df.index[int(0.85 * 400)].strftime("%Y-%m-%d")
    assert m["scaler"]["min"] == pytest.approx(df["Close"].iloc[: int(0.7 * 400)].min())  # no leakage
    assert all(np.isfinite(v) for k in ("lstm", "baseline") for v in m["metrics"][k].values())
    assert c.get("/api/model/metrics/TEST").status_code == 200
    f = c.post("/api/predict", json={"ticker": "TEST", "horizon": 5}).json()
    assert len(f["forecast"]) == 5 and f["forecast_dates"][0] > m["data_end"]
