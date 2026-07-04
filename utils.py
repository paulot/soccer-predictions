import ast
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

TEAM_TO_MANAGER: Dict[str, str] = {
    "Canada": "John Herdman",
    "Morocco": "Walid Regragui",
    "England": "Gareth Southgate",
    "Iran": "Carlos Queiroz",
    "Croatia": "Zlatko Dalić",
    "Belgium": "Roberto Martínez",
    "Netherlands": "Louis van Gaal",
    "Ecuador": "Gustavo Alfaro",
    "Japan": "Hajime Moriyasu",
    "Spain": "Luis Enrique",
}


def parse_location(loc_val: Any) -> Optional[Union[List[float], np.ndarray]]:
    if isinstance(loc_val, list) or isinstance(loc_val, np.ndarray):
        return loc_val
    if pd.isnull(loc_val):
        return None
    try:
        val = ast.literal_eval(loc_val)
        if isinstance(val, list):
            return [float(x) for x in val]
        return val
    except Exception:
        return None


def map_coordinates_to_zone(x: float, y: float) -> str:
    zone_x: int = min(int(x / 20), 5)
    zone_y: int = min(int(y / 16), 4)
    return f"Z_{zone_x}_{zone_y}"


def calculate_brier_score(prob_win: float, prob_draw: float, prob_loss: float, actual_outcome: str) -> float:
    y: np.ndarray = np.array(
        [
            1.0 if actual_outcome == "W" else 0.0,
            1.0 if actual_outcome == "D" else 0.0,
            1.0 if actual_outcome == "L" else 0.0,
        ]
    )
    p: np.ndarray = np.array([prob_win, prob_draw, prob_loss])
    return float(np.sum((p - y) ** 2))


def calculate_log_loss(prob_win: float, prob_draw: float, prob_loss: float, actual_outcome: str) -> float:
    p: Dict[str, float] = {
        "W": max(min(prob_win, 0.999), 0.001),
        "D": max(min(prob_draw, 0.999), 0.001),
        "L": max(min(prob_loss, 0.999), 0.001),
    }
    return float(-np.log(p[actual_outcome]))
