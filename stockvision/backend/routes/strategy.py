from fastapi import APIRouter

from backend.algorithms.astar import astar
from backend.schemas.models import AStarRequest
from backend.services.data_service import ApiError

router = APIRouter()


@router.post("/api/strategy/astar")
def run_astar(req: AStarRequest):
    try:
        res = astar(req.scenario, req.start_equity)
    except ValueError as e:
        raise ApiError(422, str(e))
    res["disclaimer"] = "Hypothetical, assumption-based path. A* does not predict prices; no performance is guaranteed."
    return res
