import pandas as pd
import numpy as np
from statsbombpy import sb
from mcmc_simulation import (
    build_30_zone_grid,
    map_coordinates_to_zone,
    get_zone_players,
    apply_player_modifier,
)
from utils import parse_location, calculate_brier_score, calculate_log_loss, TEAM_TO_MANAGER
from typing import Dict, List, Any, Tuple, Optional

# -------------------------------------------------------------------------
# 1. Scoring Methodologies (Imported from utils)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# 2. Full Match MCMC Simulator
# -------------------------------------------------------------------------


def simulate_full_match(
    home_team: str,
    away_team: str,
    base_matrix: pd.DataFrame,
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
    Simulates a single football match by running a sequence of possession chains,
    incorporating player skills, goalkeeper saves, defensive pressure, and manager tactics.
    Returns: (home_goals, away_goals)
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

    # Get manager profiles for both teams
    home_mgr = manager_profiles.get(team_to_manager.get(home_team, ""), {"directness": 5.0, "width": 5.0, "tempo": 5.0})
    away_mgr = manager_profiles.get(team_to_manager.get(away_team, ""), {"directness": 5.0, "width": 5.0, "tempo": 5.0})

    # Start with a goal kick (from the goalie whose team has the starting possession)
    current_team: str = np.random.choice([home_team, away_team])
    current_zone: str = "Z_0_2"  # Start at defensive penalty box (goal kick)

    for p_idx in range(num_possessions):
        chain_active: bool = True
        player_on_ball: Optional[str] = (
            (home_gk if current_team == home_team else away_gk) if (p_idx == 0 or current_zone == "Z_0_2") else None
        )

        while chain_active:
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

            # Get baseline transition probabilities
            zone_probs = base_matrix.loc[current_zone].copy()
            if zone_probs.sum() == 0:
                # Fallback: turnover
                current_team = away_team if current_team == home_team else home_team
                break

            # 1. Apply Player Modifiers
            if player_on_ball and player_on_ball in player_profiles:
                zone_probs = apply_player_modifier(zone_probs, player_profiles[player_on_ball], current_zone, zones)

            # 2. Apply Manager Tactical Modifiers (Directness and Width)
            mgr = home_mgr if current_team == home_team else away_mgr

            for zone in zones:
                end_x: int = int(zone.split("_")[1])
                end_y: int = int(zone.split("_")[2])

                # A. Directness: High directness favors long forward passes, skipping zones.
                dist_x: int = end_x - start_x
                if dist_x > 1:  # Long forward pass
                    zone_probs[zone] *= 1.0 + (mgr["directness"] - 5) * 0.12
                elif dist_x == 0 or dist_x == -1:  # Lateral or backward
                    zone_probs[zone] *= 1.0 - (mgr["directness"] - 5) * 0.04

                # B. Width: High width favors passing to wings (Y=0 or Y=4)
                if end_y in [0, 4]:
                    zone_probs[zone] *= 1.0 + (mgr["width"] - 5) * 0.08
                else:
                    zone_probs[zone] *= 1.0 - (mgr["width"] - 5) * 0.04

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

            # Re-normalize after manager & physical masking
            if zone_probs.sum() > 0:
                zone_probs = zone_probs / zone_probs.sum()
            else:
                zone_probs = pd.Series(0.0, index=zones)
                zone_probs[current_zone] = 1.0

            # Sample next zone
            next_zone = np.random.choice(base_matrix.columns, p=zone_probs.values)

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

            # 2. Turnover (dynamically determined by player accuracy + opponent defensive pressure)
            else:
                defending_team: str = away_team if current_team == home_team else home_team
                def_rate: float = team_defensive_profiles.get(defending_team, {}).get(current_zone, 0.0)
                def_factor: float = min(0.15, def_rate * 0.03)

                turnover_prob: float = 0.12 + def_factor
                if player_on_ball and player_on_ball in player_profiles:
                    turnover_prob = max(
                        0.05, min(0.30, (1.0 - player_profiles[player_on_ball]["accuracy"]) * 0.5 + 0.05 + def_factor)
                    )

                if np.random.rand() < turnover_prob:
                    current_team = defending_team
                    current_zone = next_zone
                    player_on_ball = None
                    chain_active = False
                else:
                    next_x: int = int(next_zone.split("_")[1])
                    next_y: int = int(next_zone.split("_")[2])
                    dist_x = abs(next_x - start_x)
                    dist_y = abs(next_y - start_y)

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

    return home_goals, away_goals


# -------------------------------------------------------------------------
# 3. Backtesting Pipeline (Leave-One-Out Cross-Validation)
# -------------------------------------------------------------------------


def run_loocv_backtest(match_ids: List[int], num_simulations: int = 500) -> None:
    """
    Runs a backtest over a list of Match IDs using the pre-compiled global datasets.
    """
    print(f"Starting Backtest on {len(match_ids)} matches using Advanced Global Datasets...")

    # 1. Load Global Datasets
    try:
        base_matrix: pd.DataFrame = pd.read_csv("data/global_baseline_matrix.csv", index_col=0)
        df_profiles: pd.DataFrame = pd.read_csv("data/statsbomb_player_profiles.csv")
        player_profiles: Dict[str, Dict[str, float]] = df_profiles.set_index("player").to_dict(orient="index")

        df_gk: pd.DataFrame = pd.read_csv("data/goalkeeper_profiles.csv")
        gk_profiles: Dict[str, float] = df_gk.set_index("goalkeeper")["save_ratio"].to_dict()

        df_def: pd.DataFrame = pd.read_csv("data/team_defensive_profiles.csv")
        team_defensive_profiles: Dict[str, Dict[str, float]] = {}
        for _, row in df_def.iterrows():
            t = row["team"]
            z = row["zone"]
            r = row["defensive_rate"]
            if t not in team_defensive_profiles:
                team_defensive_profiles[t] = {}
            team_defensive_profiles[t][z] = r

        # Load Manager Profiles (Dynamically generated from StatsBomb)
        df_mgr: pd.DataFrame = pd.read_csv("data/manager_profiles.csv")
        manager_profiles: Dict[str, Dict[str, float]] = df_mgr.set_index("manager").to_dict(orient="index")

        print(
            f"Successfully loaded baseline matrix, {len(player_profiles)} player profiles, "
            f"{len(gk_profiles)} goalkeepers, {len(manager_profiles)} manager profiles, and defensive profiles."
        )
    except Exception as e:
        print(f"Error loading global datasets: {e}")
        return

    zones: List[str] = build_30_zone_grid()
    results: List[Dict[str, Any]] = []

    # 2. Pre-load match events for evaluation
    print("Loading match events...")
    all_events: Dict[int, pd.DataFrame] = {}
    match_details: Dict[int, Dict[str, Any]] = {}
    for mid in match_ids:
        try:
            # Check local cache first to avoid API requests
            import os

            cache_path = f"data/raw_events/{mid}.csv"
            if os.path.exists(cache_path):
                events = pd.read_csv(cache_path)
            else:
                events = sb.events(match_id=mid)
            all_events[mid] = events

            # Extract team names and actual score
            home_team = events["team"].dropna().unique()[0]
            away_team = events["team"].dropna().unique()[1]

            # Calculate actual score from shots
            shots = events[events["type"] == "Shot"]
            home_actual = len(shots[(shots["team"] == home_team) & (shots["shot_outcome"] == "Goal")])
            away_actual = len(shots[(shots["team"] == away_team) & (shots["shot_outcome"] == "Goal")])

            if home_actual > away_actual:
                actual_outcome = "W"
            elif home_actual == away_actual:
                actual_outcome = "D"
            else:
                actual_outcome = "L"

            match_details[mid] = {
                "home_team": home_team,
                "away_team": away_team,
                "home_actual": home_actual,
                "away_actual": away_actual,
                "actual_outcome": actual_outcome,
            }
        except Exception as e:
            print(f"Skipping match {mid} due to load error: {e}")

    # 3. Run the Backtest Loop
    for target_mid in list(all_events.keys()):
        target_events = all_events[target_mid]
        home_team = match_details[target_mid]["home_team"]
        away_team = match_details[target_mid]["away_team"]

        print(f"\nEvaluating Match {target_mid}: {home_team} vs {away_team}")

        # Prepare target match passes to map player-zone occupancy for this specific game
        df_target_passes = target_events[target_events["type"] == "Pass"].copy()
        df_target_passes = df_target_passes.dropna(subset=["location", "pass_end_location", "player"])

        # Parse locations (handles string format from CSV cache)
        df_target_passes["location"] = df_target_passes["location"].apply(parse_location)
        df_target_passes["pass_end_location"] = df_target_passes["pass_end_location"].apply(parse_location)
        df_target_passes = df_target_passes.dropna(subset=["location", "pass_end_location"])

        df_target_passes["start_zone"] = df_target_passes["location"].apply(
            lambda loc: map_coordinates_to_zone(loc[0], loc[1])
        )
        df_target_passes["end_zone"] = df_target_passes["pass_end_location"].apply(
            lambda loc: map_coordinates_to_zone(loc[0], loc[1])
        )

        # Build player-to-team mapping for this match
        player_to_team = target_events.dropna(subset=["player", "team"]).set_index("player")["team"].to_dict()

        # Calculate dynamic tempo based on manager profiles (baseline 100 possessions)
        home_mgr = manager_profiles.get(TEAM_TO_MANAGER.get(home_team, ""), {"tempo": 5.0})
        away_mgr = manager_profiles.get(TEAM_TO_MANAGER.get(away_team, ""), {"tempo": 5.0})
        tempo_factor = (
            1.0 + (home_mgr["tempo"] + away_mgr["tempo"] - 10) * 0.05
        )  # e.g. both 7 tempo -> 1.20 (120 possessions)
        dynamic_possessions = int(100 * tempo_factor)

        # Run MCMC Simulation
        home_wins = 0
        draws = 0
        away_wins = 0

        print(f"Simulating match {num_simulations} times ({dynamic_possessions} possessions/game)...")
        for _ in range(num_simulations):
            h_goals, a_goals = simulate_full_match(
                home_team,
                away_team,
                base_matrix,
                df_target_passes,
                player_profiles,
                gk_profiles,
                team_defensive_profiles,
                manager_profiles,
                TEAM_TO_MANAGER,
                player_to_team,
                zones,
                num_possessions=dynamic_possessions,
            )
            if h_goals > a_goals:
                home_wins += 1
            elif h_goals == a_goals:
                draws += 1
            else:
                away_wins += 1

        prob_win = home_wins / num_simulations
        prob_draw = draws / num_simulations
        prob_loss = away_wins / num_simulations

        actual_outcome = match_details[target_mid]["actual_outcome"]

        # Scoring
        brier = calculate_brier_score(prob_win, prob_draw, prob_loss, actual_outcome)
        logloss = calculate_log_loss(prob_win, prob_draw, prob_loss, actual_outcome)

        # Check if the highest probability matches the actual outcome (Accuracy)
        predicted_outcome = (
            "W" if prob_win > prob_draw and prob_win > prob_loss else ("D" if prob_draw > prob_loss else "L")
        )
        is_correct = 1 if predicted_outcome == actual_outcome else 0

        print(f"Predictions: Win={prob_win:.2%}, Draw={prob_draw:.2%}, Loss={prob_loss:.2%}")
        print(
            f"Actual Outcome: {actual_outcome} (Score: {match_details[target_mid]['home_actual']}-{match_details[target_mid]['away_actual']})"  # noqa: E501
        )
        print(f"Brier Score: {brier:.4f} | Log Loss: {logloss:.4f} | Correct: {is_correct}")

        results.append({"match_id": target_mid, "brier": brier, "log_loss": logloss, "correct": is_correct})

    df_results = pd.DataFrame(results)

    print("\n==================================================")
    print("BACKTEST RESULTS (GLOBAL DATA SUMMARY)")
    print("==================================================")
    print(f"Average Brier Score: {df_results['brier'].mean():.4f}  (Lower is better, benchmark is ~0.66)")
    print(f"Average Log Loss:    {df_results['log_loss'].mean():.4f} (Lower is better, benchmark is ~1.098)")
    print(f"Overall Accuracy:    {df_results['correct'].mean():.2%}")
    print("==================================================")


if __name__ == "__main__":
    # Dynamically fetch 5 valid match IDs from the 2022 World Cup to evaluate
    print("Querying valid World Cup 2022 Match IDs...")
    competitions = sb.competitions()
    wc_2022 = competitions[
        (competitions["competition_name"] == "FIFA World Cup") & (competitions["season_name"] == "2022")
    ].iloc[0]

    matches = sb.matches(competition_id=wc_2022["competition_id"], season_id=wc_2022["season_id"])

    # We select 5 matches
    sample_match_ids = matches["match_id"].head(5).tolist()

    # Run the backtest using the pre-compiled global baseline matrix and player profiles
    run_loocv_backtest(sample_match_ids, num_simulations=500)
