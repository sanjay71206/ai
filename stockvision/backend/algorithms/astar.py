"""A* over hypothetical portfolio-allocation states (equity % in steps of 10).
This does NOT predict prices; it searches a defined strategy state space."""
import heapq
import math

SCENARIOS = {
    "c": {"name": "Conservative", "target": 30, "risk": 2.0},
    "b": {"name": "Balanced", "target": 60, "risk": 1.0},
    "g": {"name": "Growth", "target": 90, "risk": 0.3},
}
ALIASES = {"conservative": "c", "balanced": "b", "growth": "g"}
MOVES = (-20, -10, 10, 20)


def transition_cost(a: int, b: int, risk: float) -> float:
    """1 + trade size penalty + risk-aversion * exposure^2 at the new state."""
    return 1 + abs(b - a) / 20 + risk * (b / 100) ** 2 * 2


def heuristic(e: int, target: int) -> int:
    """Admissible: every move costs >= 1 and shifts equity by <= 20 points."""
    return math.ceil(abs(e - target) / 20)


def astar(scenario: str = "b", start: int = 50) -> dict:
    key = ALIASES.get(scenario, scenario)
    if key not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}")
    sc = SCENARIOS[key]
    target, risk = sc["target"], sc["risk"]
    g = {start: 0.0}
    parent: dict = {}
    heap = [(heuristic(start, target), 0.0, start)]
    closed: list = []
    expanded: list = []
    log: list = []
    while heap:
        _, gc, e = heapq.heappop(heap)
        if e in closed:
            continue
        closed.append(e)
        log.append(f"Expand {e}% equity · g={gc:.2f} h={heuristic(e, target)} f={gc + heuristic(e, target):.2f}")
        if e == target:
            path, cur = [], e
            while True:
                path.insert(0, cur)
                if cur not in parent:
                    break
                cur = parent[cur]
            return {"scenario": key, "name": sc["name"], "target": target, "start": start,
                    "path": path, "cost": round(gc, 6), "log": log, "closed": closed, "expanded": expanded}
        for d in MOVES:
            n = e + d
            if n < 0 or n > 100 or n in closed:
                continue
            c = transition_cost(e, n, risk)
            expanded.append([e, n, round(c, 4)])
            if n not in g or gc + c < g[n]:
                g[n] = gc + c
                parent[n] = e
                heapq.heappush(heap, (g[n] + heuristic(n, target), g[n], n))
    raise ValueError("No path found")
