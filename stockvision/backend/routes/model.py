from fastapi import APIRouter

from backend.ml import lstm_model
from backend.schemas.models import PredictRequest, TrainRequest
from backend.services.data_service import ApiError, fetch_history, validate_ticker

router = APIRouter()


@router.post("/api/model/train")
def train(req: TrainRequest):
    df = fetch_history(req.ticker, "5y")
    try:
        return lstm_model.train_model(req.ticker, df, req.lookback, req.epochs)
    except ApiError:
        raise
    except Exception as e:
        raise ApiError(500, f"Model training failed: {str(e)[:200]}")


@router.get("/api/model/metrics/{ticker}")
def metrics(ticker: str):
    return lstm_model.read_metrics(validate_ticker(ticker))


@router.post("/api/predict")
def predict(req: PredictRequest):
    lstm_model.read_metrics(req.ticker)  # 404 early if untrained
    df = fetch_history(req.ticker, "5y")
    try:
        return lstm_model.forecast(req.ticker, df, req.horizon)
    except ApiError:
        raise
    except Exception as e:
        raise ApiError(500, f"Forecast failed: {str(e)[:200]}")
