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

    # Start with a kickoff (stochastically given to one team)
    current_team: str = np.random.choice([home_team, away_team])
    current_zone: str = "Z_2_2"  # Start in central midfield

    for p_idx in range(num_possessions):
        chain_active: bool = True
        # Initialize sequence history buffer for the possession: [prev_1, prev_2]
        # Each entry is (zone_x, zone_y, success)
        history: List[Tuple[int, int, int]] = [(-1, -1, -1), (-1, -1, -1)]
        possession_duration: float = 0.0
        pass_sequence_index: int = 0
        possession_directions: List[int] = []

        time_ratio: float = p_idx / num_possessions

        while chain_active:
            # Prevent infinite loops (e.g. if turnover probability is 0)
            if pass_sequence_index >= 50:
                current_team = away_team if current_team == home_team else home_team
                chain_active = False
                break

            # Get players in this zone
            zone_players = get_zone_players(df_events, current_zone)

            # Filter for players who actually play for the team currently in possession
            team_zone_players = {p: w for p, w in zone_players.items() if player_to_team.get(p) == current_team}

            player_on_ball: Optional[str] = None
            if team_zone_players:
                # Re-normalize weights
                total_w = sum(team_zone_players.values())
                team_zone_players = {p: w / total_w for p, w in team_zone_players.items()}
                player_on_ball = np.random.choice(list(team_zone_players.keys()), p=list(team_zone_players.values()))

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
            )

            if zone_probs.sum() == 0:
                # Fallback: turnover
                current_team = away_team if current_team == home_team else home_team
                break

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

                # Goal ends possession, opponent kicks off from center
                current_team = away_team if current_team == home_team else home_team
                current_zone = "Z_2_2"
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
                    chain_active = False
                else:
                    # Update history buffer with this successful pass
                    curr_x: int = int(current_zone.split("_")[1])
                    curr_y: int = int(current_zone.split("_")[2])

                    next_x: int = int(next_zone.split("_")[1])
                    dx: int = next_x - curr_x
                    direction: int = 1 if dx > 0 else (-1 if dx < 0 else 0)
                    possession_directions.append(direction)

                    history = [(curr_x, curr_y, 1), history[0]]
                    possession_duration += 3.0
                    pass_sequence_index += 1

                    current_zone = next_zone

    return home_goals, away_goals
