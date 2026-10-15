from datetime import datetime, timezone
from importlib.util import find_spec

from fastapi import APIRouter

router = APIRouter()


@router.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat(),
            "tensorflow": find_spec("tensorflow") is not None, "yfinance": find_spec("yfinance") is not None}
