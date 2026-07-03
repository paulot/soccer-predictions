import ast
import numpy as np
import pandas as pd

TEAM_TO_MANAGER = {
    "Canada": "John Herdman",
    "Morocco": "Walid Regragui",
    "England": "Gareth Southgate",
    "Iran": "Carlos Queiroz",
    "Croatia": "Zlatko Dalić",
    "Belgium": "Roberto Martínez",
    "Netherlands": "Louis van Gaal",
    "Ecuador": "Gustavo Alfaro",
    "Japan": "Hajime Moriyasu",
    "Spain": "Luis Enrique"
}

def parse_location(loc_val):
    if pd.isnull(loc_val):
        return None
    if isinstance(loc_val, list) or isinstance(loc_val, np.ndarray):
        return loc_val
    try:
        return ast.literal_eval(loc_val)
    except:
        return None

def map_coordinates_to_zone(x, y):
    zone_x = min(int(x / 20), 5)
    zone_y = min(int(y / 16), 4)
    return f"Z_{zone_x}_{zone_y}"

def calculate_brier_score(prob_win, prob_draw, prob_loss, actual_outcome):
    y = np.array([1.0 if actual_outcome == 'W' else 0.0,
                  1.0 if actual_outcome == 'D' else 0.0,
                  1.0 if actual_outcome == 'L' else 0.0])
    p = np.array([prob_win, prob_draw, prob_loss])
    return np.sum((p - y) ** 2)

def calculate_log_loss(prob_win, prob_draw, prob_loss, actual_outcome):
    p = {
        'W': max(min(prob_win, 0.999), 0.001),
        'D': max(min(prob_draw, 0.999), 0.001),
        'L': max(min(prob_loss, 0.999), 0.001)
    }
    return -np.log(p[actual_outcome])
