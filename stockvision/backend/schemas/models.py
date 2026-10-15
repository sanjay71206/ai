from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.services.data_service import TICKER_RE


class _T(BaseModel):
    ticker: str

    @field_validator("ticker")
    @classmethod
    def _chk(cls, v):
        if not TICKER_RE.match(v.strip()):
            raise ValueError("Invalid ticker symbol")
        return v.strip().upper()


class TrainRequest(_T):
    lookback: int = Field(20, ge=5, le=120)
    epochs: int = Field(20, ge=1, le=100)


class PredictRequest(_T):
    horizon: int = Field(10, ge=1, le=60)


class AStarRequest(BaseModel):
    scenario: Literal["c", "b", "g", "conservative", "balanced", "growth"] = "b"
    start_equity: int = Field(50, ge=0, le=100)

    @field_validator("start_equity")
    @classmethod
    def _step(cls, v):
        if v % 10:
            raise ValueError("start_equity must be a multiple of 10")
        return v
