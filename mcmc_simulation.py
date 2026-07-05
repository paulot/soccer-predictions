import os
import numpy as np
import pandas as pd
from utils import map_coordinates_to_zone
from typing import List, Dict, Optional
from ml_model.corners.models import CornerRoutineXGB, CornerOutcomeXGB

try:
    from statsbombpy import sb
except ImportError:
    print("Please install statsbombpy first: pip install statsbombpy")
    exit(1)


def build_30_zone_grid() -> List[str]:
    """Creates a 6x5 grid representing the 30 zones on a pitch."""
    zones: List[str] = []
    for x in range(6):
        for y in range(5):
            zones.append(f"Z_{x}_{y}")
    return zones


def fetch_real_statsbomb_data() -> pd.DataFrame:
    """
    Fetches real event data from StatsBomb Open Data.
    For this example, we use the 2022 World Cup Final: Argentina vs France (Match ID: 3869685)
    """
    print("Fetching 2022 World Cup Final (Argentina vs France) events from StatsBomb...")
    events: pd.DataFrame = sb.events(match_id=3869685)

    # Filter for passes to build our spatial transition matrix
    passes = events[events["type"] == "Pass"].copy()
    passes = passes.dropna(subset=["location", "pass_end_location"])

    # Map raw coordinates to our discrete zones
    passes["start_zone"] = passes["location"].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
    passes["end_zone"] = passes["pass_end_location"].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))

    # In StatsBomb, a null 'pass_outcome' means the pass was successful
    passes["event_type"] = passes["pass_outcome"].apply(lambda outcome: "Turnover" if pd.notnull(outcome) else "Pass")

    return passes


def calculate_player_profiles(df_events: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculates player-specific metrics from the event data:
    - progressive_ratio: proportion of successful passes that move the ball forward (increase in X)
    - pass_accuracy: ratio of successful passes to total passes
    """
    profiles: Dict[str, Dict[str, float]] = {}
    players: np.ndarray = df_events["player"].dropna().unique()

    for player in players:
        player_passes = df_events[df_events["player"] == player]
        total_attempts = len(player_passes)
        if total_attempts < 5:  # Skip players with too few passes
            continue

        successful = player_passes[player_passes["event_type"] == "Pass"]
        accuracy = len(successful) / total_attempts

        # Calculate progressive ratio (passes moving the ball forward on the X-axis)
        forward_passes = 0
        for _, row in successful.iterrows():
            start_x = int(row["start_zone"].split("_")[1])
            end_x = int(row["end_zone"].split("_")[1])
            if end_x > start_x:
                forward_passes += 1

        prog_ratio = forward_passes / len(successful) if len(successful) > 0 else 0.0

        profiles[player] = {
            "accuracy": float(accuracy),
            "progressive_ratio": float(prog_ratio),
            "total_passes": float(total_attempts),
        }

    return profiles


def get_zone_players(df_events: pd.DataFrame, zone: str) -> Dict[str, float]:
    """Returns players who historically made passes from this zone, with their frequencies."""
    zone_passes = df_events[df_events["start_zone"] == zone]
    if zone_passes.empty:
        return {}
    return zone_passes["player"].value_counts(normalize=True).to_dict()


def build_baseline_transition_matrix(df_events: pd.DataFrame, zones: List[str]) -> pd.DataFrame:
    """
    Builds the baseline transition matrix P(End_Zone | Start_Zone) using real event frequencies.
    """
    matrix: pd.DataFrame = pd.DataFrame(0.0, index=zones, columns=zones)

    # Count successful passes between zones
    successful_passes = df_events[df_events["event_type"] == "Pass"]
    transitions = successful_passes.groupby(["start_zone", "end_zone"]).size().reset_index(name="count")

    for _, row in transitions.iterrows():
        matrix.at[row["start_zone"], row["end_zone"]] = float(row["count"])

    # Row normalize so probabilities sum to 1.0
    row_sums = matrix.sum(axis=1)
    matrix = matrix.div(row_sums, axis=0)
    matrix = matrix.fillna(0.0)

    return matrix


def apply_player_modifier(
    row_probs: pd.Series, player_profile: Dict[str, float], start_zone: str, zones: List[str]
) -> pd.Series:
    """
    Adjusts the transition probabilities of a zone based on the player's progressive passing profile.
    """
    start_x = int(start_zone.split("_")[1])

    # Scale forward passes based on progressive_ratio (baseline average is ~0.33)
    prog_multiplier = player_profile["progressive_ratio"] / 0.33

    modified_probs = row_probs.copy()
    for zone in zones:
        end_x = int(zone.split("_")[1])
        if end_x > start_x:
            modified_probs[zone] *= prog_multiplier

    # Re-normalize the probabilities
    if modified_probs.sum() > 0:
        modified_probs = modified_probs / modified_probs.sum()

    return modified_probs


def simulate_mcmc_possession_chain(
    start_zone: str,
    base_matrix: pd.DataFrame,
    df_events: pd.DataFrame,
    player_profiles: Dict[str, Dict[str, float]],
    zones: List[str],
    max_steps: int = 10,
    routine_model: Optional[CornerRoutineXGB] = None,
    outcome_model: Optional[CornerOutcomeXGB] = None,
) -> List[str]:
    """
    Simulates a possession chain, dynamically selecting the player on the ball
    based on the zone and applying their specific pass modifiers.
    """
    current_zone: str = start_zone
    chain: List[str] = [current_zone]

    print(f"\n--- Simulating Player-Modified Possession Chain starting in {start_zone} ---")
    for step in range(max_steps):
        # 1. Determine who has the ball in this zone based on historical match data
        zone_players = get_zone_players(df_events, current_zone)
        if not zone_players:
            print(f"Step {step+1}: Chain ended in {current_zone} (No player data for this zone)")
            break

        # Select player stochastically based on who played in this zone
        player_on_ball = np.random.choice(list(zone_players.keys()), p=list(zone_players.values()))

        # Check if Corner Kick Scenario (Z_5_0 is left corner flag, Z_5_4 is right corner flag)
        if current_zone in ["Z_5_0", "Z_5_4"] and routine_model is not None and outcome_model is not None:
            print(f"Step {step+1}: 🚩 Corner Kick Scenario in {current_zone}! Leveraged trained Corner Models.")
            is_right = 1 if current_zone == "Z_5_4" else 0
            profile = player_profiles.get(player_on_ball, {}) if player_on_ball else {}

            corner_features = pd.DataFrame([{
                "is_right_corner": is_right,
                "time_ratio": 0.5,
                "score_differential": 0,
                "is_home_team": 1,
                "inswinging": 1,
                "taker_accuracy": profile.get("accuracy", 0.78),
                "taker_key_pass_ratio": profile.get("progressive_ratio", 0.22),
                "team_directness": 5.0,
                "team_width": 5.0,
                "opp_gk_save_ratio": 0.70,
                "opp_def_rate": 0.15,
                "under_pressure": 0,
                "prev_corner_routine_in_match": -1,
                "corner_cluster_density": 1,
                "aerial_height_advantage": 0.0,
                "goalkeeper_line_command": 0.70,
                "taker_corner_assist_rate": 0.12,
                "delivery_curve_match": 1,
            }])

            # Stage 1: Predict Routine
            r_probs = routine_model.predict_proba(corner_features)[0]
            routine = np.random.choice([0, 1, 2], p=r_probs)

            if routine == 0:
                next_zone = "Z_5_2"
                r_name = "Direct Central Box (Z_5_2)"
            elif routine == 1:
                next_zone = np.random.choice(["Z_5_1", "Z_5_3"])
                r_name = f"Post Cross ({next_zone})"
            else:
                next_zone = np.random.choice(["Z_4_0", "Z_4_4"])
                r_name = f"Short Corner ({next_zone})"

            print(f"  -> Stage 1 Routine Prediction: {r_name} [Probs: {r_probs.round(2)}]")

            # Stage 2: Predict Outcome (ONLY for Direct Crosses!)
            if routine == 2:
                print(f"  -> Short Corner selected! Outcome model bypassed; play continues as normal open play in {next_zone}.")
                current_zone = next_zone
                chain.append(next_zone)
                continue
            else:
                o_prob = outcome_model.predict_proba(corner_features)[0]
                outcome = 1 if np.random.rand() < o_prob else 0
                outcome_str = "ATTACKING SUCCESS (Shot/Goal Opportunity in Box!)" if outcome == 1 else "DEFENSIVE SUCCESS (Clearance/Catch)"
                print(f"  -> Stage 2 Outcome Prediction: {outcome_str} [Success Prob: {o_prob:.1%}]")
                chain.append(next_zone)
                break

        # 2. Get baseline probabilities for this zone
        zone_probs = base_matrix.loc[current_zone].copy()
        if zone_probs.sum() == 0:
            print(f"Step {step+1}: Chain ended in {current_zone} (No transition data)")
            break

        # 3. Apply player modifier if they have a profile
        if player_on_ball in player_profiles:
            profile = player_profiles[player_on_ball]
            zone_probs = apply_player_modifier(zone_probs, profile, current_zone, zones)
            player_info = (
                f"[{player_on_ball} | Acc: {profile['accuracy']:.2f}, Prog: {profile['progressive_ratio']:.2f}]"
            )
        else:
            player_info = f"[{player_on_ball} (No Profile)]"

        # 4. Sample the next zone
        next_zone = np.random.choice(base_matrix.columns, p=zone_probs.values)
        chain.append(next_zone)

        print(f"Step {step+1}: {player_info} in {current_zone} -> passes to {next_zone}")
        current_zone = next_zone

        # End if ball enters the penalty box (Z_5_1, Z_5_2, Z_5_3)
        if current_zone in ["Z_5_1", "Z_5_2", "Z_5_3"]:
            print("Outcome: Ball entered the attacking penalty box! (Shot opportunity)")
            break

    return chain


if __name__ == "__main__":
    zones_list = build_30_zone_grid()

    # 1. Fetch real match data
    try:
        df = fetch_real_statsbomb_data()
        print(f"Successfully loaded {len(df)} passes from StatsBomb.")
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        exit(1)

    # 2. Build Profiles and Matrices
    profiles_dict = calculate_player_profiles(df)
    matrix_df = build_baseline_transition_matrix(df, zones_list)

    # Print a few top player profiles for demonstration
    print("\nSample Player Profiles (World Cup Final):")
    for p_name in [
        "Lionel Andrés Messi Cuccittini",
        "Enzo Jeremías Fernández",
        "Antoine Griezmann",
        "Aurélien Djani Tchouaméni",
    ]:
        if p_name in profiles_dict:
            prof = profiles_dict[p_name]
            print(
                f"- {p_name}: Accuracy={prof['accuracy']:.2f}, Progressive Ratio={prof['progressive_ratio']:.2f} (Total Passes: {prof['total_passes']})"  # noqa: E501
            )

    # 3. Load Trained Corner Models if available
    routine_model, outcome_model = None, None
    try:
        if os.path.exists("data/models/corner_routine_xgb_production.pkl"):
            routine_model = CornerRoutineXGB.load("data/models/corner_routine_xgb_production.pkl")
            outcome_model = CornerOutcomeXGB.load("data/models/corner_outcome_xgb_production.pkl")
        elif os.path.exists("data/models/corner_routine_xgb_iteration.pkl"):
            routine_model = CornerRoutineXGB.load("data/models/corner_routine_xgb_iteration.pkl")
            outcome_model = CornerOutcomeXGB.load("data/models/corner_outcome_xgb_iteration.pkl")
        if routine_model and outcome_model:
            print("\nSuccessfully loaded trained Corner Routine and Outcome models.")
    except Exception as e:
        print(f"\nNotice: Could not load corner models ({e}). Corner scenarios will use baseline transitions.")

    # 4. Run MCMC Simulations with Player Modifiers and Corner Models
    simulate_mcmc_possession_chain("Z_2_2", matrix_df, df, profiles_dict, zones_list, routine_model=routine_model, outcome_model=outcome_model)
    simulate_mcmc_possession_chain("Z_3_2", matrix_df, df, profiles_dict, zones_list, routine_model=routine_model, outcome_model=outcome_model)
    simulate_mcmc_possession_chain("Z_5_4", matrix_df, df, profiles_dict, zones_list, routine_model=routine_model, outcome_model=outcome_model)
    simulate_mcmc_possession_chain("Z_5_0", matrix_df, df, profiles_dict, zones_list, routine_model=routine_model, outcome_model=outcome_model)
