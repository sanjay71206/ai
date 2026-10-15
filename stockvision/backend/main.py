"""StockVision AI API.  Run from project root:  uvicorn backend.main:app --reload"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.routes import health, model, stock, strategy
from backend.services.data_service import ApiError

load_dotenv()
app = FastAPI(title="StockVision AI", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])
for r in (health, stock, model, strategy):
    app.include_router(r.router)


@app.exception_handler(ApiError)
async def _api_error(_, exc: ApiError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.exception_handler(RequestValidationError)
async def _validation(_, exc: RequestValidationError):
    e = exc.errors()[0]
    loc = ".".join(str(x) for x in e["loc"] if x not in ("body", "query", "path"))
    return JSONResponse(status_code=422, content={"detail": f"{loc}: {e['msg']}"})


INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(INDEX)
