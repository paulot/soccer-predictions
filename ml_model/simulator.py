import numpy as np
import pandas as pd
from mcmc_simulation import get_zone_players
from typing import Dict, List, Any, Tuple, Optional


def simulate_full_match(
    home_team: str,
    away_team: str,
    transition_model: Any,
    df_events: pd.DataFrame,
    player_profiles: Dict[str, Dict[str, float]],
    gk_profiles: Dict[str, float],
    team_defensive_profiles: Dict[str, Dict[str, float]],
    manager_profiles: Dict[str, Dict[str, float]],
    team_to_manager: Dict[str, str],
    player_to_team: Dict[str, str],
    zones: List[str],
    num_possessions: int = 100,
) -> Tuple[int, int]:
    """
    Simulates a single football match using a modular TransitionModel (Heuristic or ML).
    """
    home_goals: int = 0
    away_goals: int = 0

    # Identify goalkeepers for both teams
    home_gk: Optional[str] = None
    away_gk: Optional[str] = None
    for player, team in player_to_team.items():
        if player in gk_profiles:
            if team == home_team:
                home_gk = player
            elif team == away_team:
                away_gk = player

    # Start with a goal kick (from the goalie whose team has the starting possession)
    current_team: str = np.random.choice([home_team, away_team])
    current_zone: str = "Z_0_2"  # Start at defensive penalty box (goal kick)

    for p_idx in range(num_possessions):
        chain_active: bool = True
        # Initialize sequence history buffer for the possession: [prev_1, prev_2]
        # Each entry is (zone_x, zone_y, success)
        history: List[Tuple[int, int, int]] = [(-1, -1, -1), (-1, -1, -1)]
        possession_duration: float = 0.0
        pass_sequence_index: int = 0
        possession_directions: List[int] = []
        player_on_ball: Optional[str] = (
            (home_gk if current_team == home_team else away_gk) if (p_idx == 0 or current_zone == "Z_0_2") else None
        )

        time_ratio: float = p_idx / num_possessions

        while chain_active:
            # Prevent infinite loops (e.g. if turnover probability is 0)
            if pass_sequence_index >= 50:
                current_team = away_team if current_team == home_team else home_team
                chain_active = False
                break

            start_x: int = int(current_zone.split("_")[1])
            start_y: int = int(current_zone.split("_")[2])

            # Get players in this zone
            zone_players = get_zone_players(df_events, current_zone)

            # Filter for players who actually play for the team currently in possession
            team_zone_players = {p: w for p, w in zone_players.items() if player_to_team.get(p) == current_team}

            if player_on_ball is None or player_on_ball not in team_zone_players:
                if team_zone_players:
                    total_w = sum(team_zone_players.values())
                    weights = {p: w / total_w for p, w in team_zone_players.items()}
                    player_on_ball = np.random.choice(list(weights.keys()), p=list(weights.values()))
                else:
                    player_on_ball = None

            # Calculate score differential from the perspective of the team in possession
            score_differential: int = home_goals - away_goals if current_team == home_team else away_goals - home_goals

            # Determine defending team and opponent defensive rate in current zone for under_pressure sampling
            defending_team: str = away_team if current_team == home_team else home_team
            opp_def_rate: float = team_defensive_profiles.get(defending_team, {}).get(current_zone, 0.0)
            under_pressure: int = 1 if np.random.rand() < opp_def_rate else 0

            # Get transition probabilities from the modular model
            zone_probs = transition_model.get_transition_probabilities(
                current_zone,
                player_on_ball,
                player_profiles,
                manager_profiles,
                team_to_manager,
                player_to_team,
                home_team,
                away_team,
                zones,
                history=history,
                score_differential=score_differential,
                possession_duration=possession_duration,
                pass_sequence_index=pass_sequence_index,
                possession_directions=possession_directions,
                time_ratio=time_ratio,
                under_pressure=under_pressure,
            ).copy()

            # --- FSM RULE 1 & 4: Teammate Occupancy Masking & Kinematic Distance Bounding ---
            for target_zone in zones:
                tx, ty = int(target_zone.split("_")[1]), int(target_zone.split("_")[2])
                dist = np.sqrt((tx - start_x) ** 2 + (ty - start_y) ** 2)
                if dist > 3.5:
                    zone_probs[target_zone] = 0.0
                    continue
                target_players = get_zone_players(df_events, target_zone)
                teammate_count = sum(1 for p in target_players.keys() if player_to_team.get(p) == current_team)
                if teammate_count == 0 and target_zone != current_zone:
                    zone_probs[target_zone] = 0.0

            if zone_probs.sum() > 0:
                zone_probs = zone_probs / zone_probs.sum()
            else:
                zone_probs = pd.Series(0.0, index=zones)
                zone_probs[current_zone] = 1.0

            # Sample next zone
            next_zone = np.random.choice(zone_probs.index, p=zone_probs.values)

            # --- TRANSITION LOGIC ---
            # 1. Shot Opportunity (Entering Z_5_x)
            if next_zone.startswith("Z_5_"):
                # Determine shooter's conversion rate (fallback to 10%)
                conversion: float = 0.10
                if player_on_ball and player_on_ball in player_profiles:
                    conversion = player_profiles[player_on_ball].get("shot_conversion", 0.10)
                    if conversion == 0.0:
                        conversion = 0.10

                # Determine opposing goalkeeper's save multiplier
                opp_gk = away_gk if current_team == home_team else home_gk
                save_ratio = gk_profiles.get(opp_gk or "", 0.70)
                gk_multiplier: float = max(0.5, min(1.5, (1.0 - save_ratio) / 0.30))

                final_conversion_rate: float = conversion * gk_multiplier

                if np.random.rand() < final_conversion_rate:
                    if current_team == home_team:
                        home_goals += 1
                    else:
                        away_goals += 1

                # Goal ends possession, opponent restarts with a goal kick from their goalie
                current_team = away_team if current_team == home_team else home_team
                current_zone = "Z_0_2"
                player_on_ball = home_gk if current_team == home_team else away_gk
                chain_active = False

            # 2. Turnover (dynamically determined by the modular model)
            else:
                turnover_prob = transition_model.get_turnover_probability(
                    current_zone,
                    player_on_ball,
                    player_profiles,
                    team_defensive_profiles,
                    player_to_team,
                    home_team,
                    away_team,
                    history=history,
                    score_differential=score_differential,
                    possession_duration=possession_duration,
                    pass_sequence_index=pass_sequence_index,
                    next_zone=next_zone,
                    possession_directions=possession_directions,
                    time_ratio=time_ratio,
                    under_pressure=under_pressure,
                )

                if np.random.rand() < turnover_prob:
                    current_team = away_team if current_team == home_team else home_team
                    current_zone = next_zone
                    player_on_ball = None
                    chain_active = False
                else:
                    # Update history buffer with this successful transition
                    next_x: int = int(next_zone.split("_")[1])
                    next_y: int = int(next_zone.split("_")[2])
                    dist_x: int = abs(next_x - start_x)
                    dist_y: int = abs(next_y - start_y)
                    dx: int = next_x - start_x
                    direction: int = 1 if dx > 0 else (-1 if dx < 0 else 0)
                    possession_directions.append(direction)

                    # --- FSM RULE 2: Classify non-shot action as Hold, Carry, or Pass ---
                    if next_zone == current_zone:
                        # Hold / Delay: same player retains ball
                        pass
                    elif dist_x <= 1 and dist_y <= 1 and np.random.rand() < 0.35:
                        # Carry / Run: same player moves with ball
                        current_zone = next_zone
                    else:
                        # Pass: transfer to teammate in next_zone
                        current_zone = next_zone
                        target_players = get_zone_players(df_events, next_zone)
                        team_targets = {
                            p: w
                            for p, w in target_players.items()
                            if player_to_team.get(p) == current_team and p != player_on_ball
                        }
                        if not team_targets:
                            team_targets = {
                                p: w for p, w in target_players.items() if player_to_team.get(p) == current_team
                            }
                        if team_targets:
                            total_w = sum(team_targets.values())
                            weights = {p: w / total_w for p, w in team_targets.items()}
                            player_on_ball = np.random.choice(list(weights.keys()), p=list(weights.values()))
                        else:
                            player_on_ball = None

                    history = [(start_x, start_y, 1), history[0]]
                    possession_duration += 3.0
                    pass_sequence_index += 1

    return home_goals, away_goals
