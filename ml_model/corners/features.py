import os
import click
import pandas as pd
import numpy as np
from utils import parse_location, map_coordinates_to_zone, TEAM_TO_MANAGER
from typing import Dict, List, Any, Optional, Tuple


def extract_corner_features(mode: str = "iteration") -> pd.DataFrame:
    """
    Extracts tabular features and target labels specifically for corner kicks from raw event files.
    """
    print(f"Extracting features for Corner Kicks (Mode: {mode.upper()})...")

    # 1. Load global profiles for joining
    player_profiles: Dict[str, Dict[str, float]] = {}
    gk_profiles: Dict[str, Dict[str, float]] = {}
    def_profiles: Dict[Tuple[str, str], float] = {}
    mgr_profiles: Dict[str, Dict[str, float]] = {}

    try:
        if os.path.exists("data/statsbomb_player_profiles.csv"):
            df_players = pd.read_csv("data/statsbomb_player_profiles.csv")
            player_profiles = df_players.set_index("player").to_dict(orient="index")

        if os.path.exists("data/goalkeeper_profiles.csv"):
            df_gk = pd.read_csv("data/goalkeeper_profiles.csv")
            gk_profiles = df_gk.set_index("goalkeeper").to_dict(orient="index")

        if os.path.exists("data/team_defensive_profiles.csv"):
            df_def = pd.read_csv("data/team_defensive_profiles.csv")
            for _, row in df_def.iterrows():
                def_profiles[(str(row["team"]), str(row["zone"]))] = float(row["defensive_rate"])

        if os.path.exists("data/manager_profiles.csv"):
            df_mgr = pd.read_csv("data/manager_profiles.csv")
            mgr_profiles = df_mgr.set_index("manager").to_dict(orient="index")
    except Exception as e:
        print(f"  Warning loading profiles: {e}. Default profile values will be used.")

    raw_dir: str = "data/raw_events"
    if not os.path.exists(raw_dir):
        print(f"Raw events directory {raw_dir} not found. Please run download_data.py first.")
        return pd.DataFrame()

    match_files: List[str] = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
    match_files.sort(key=lambda f: int(f.split('.')[0]) if f.split('.')[0].isdigit() else 0)

    if mode == "iteration":
        match_files = match_files[:50]
        print(f"  Iteration Mode: Limiting corner feature extraction to first {len(match_files)} matches.")

    dataset: List[Dict[str, Any]] = []
    taker_corners: Dict[str, int] = {}
    taker_assists: Dict[str, int] = {}
    team_hist_routines: Dict[str, List[int]] = {}

    for idx, fname in enumerate(match_files):
        fpath = os.path.join(raw_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue

        if df.empty or "type" not in df.columns:
            continue

        # Track match score differential
        home_team: Optional[str] = None
        away_team: Optional[str] = None
        home_goals: int = 0
        away_goals: int = 0

        # Identify home/away team from Starting XI or first events
        for _, row in df.iterrows():
            if pd.notna(row.get("team")):
                if home_team is None:
                    home_team = str(row["team"])
                elif away_team is None and str(row["team"]) != home_team:
                    away_team = str(row["team"])
                    break

        team_prev_routine: Dict[str, int] = {}
        team_corner_times: Dict[str, List[float]] = {}
        team_aerial_wins: Dict[str, int] = {}
        team_aerial_total: Dict[str, int] = {}
        opp_gk_claims: int = 0
        opp_gk_punches: int = 0

        # Pre-compute match aerial duels and goalie actions
        if "type" in df.columns:
            for _, m_row in df.iterrows():
                m_type = str(m_row.get("type", ""))
                m_team = str(m_row.get("team", ""))
                if m_type == "Duel":
                    d_type = str(m_row.get("duel_type", ""))
                    if "Aerial" in d_type:
                        team_aerial_total[m_team] = team_aerial_total.get(m_team, 0) + 1
                        if d_type == "Aerial Won":
                            team_aerial_wins[m_team] = team_aerial_wins.get(m_team, 0) + 1
                elif m_type == "Goal Keeper":
                    gk_type = str(m_row.get("goalkeeper_type", ""))
                    if gk_type in ["Claim", "Catch", "Collected"]:
                        opp_gk_claims += 1
                    elif gk_type == "Punch":
                        opp_gk_punches += 1

        # Filter for corner kick pass events
        is_corner = (df["type"] == "Pass") & (
            (df.get("play_pattern") == "From Corner") | (df.get("pass_type") == "Corner")
        )
        corner_indices = df[is_corner].index

        for c_idx in corner_indices:
            row = df.loc[c_idx]
            corner_team = str(row.get("team", ""))
            if not corner_team:
                continue

            # Check if right corner (y > 40)
            start_loc = parse_location(row.get("location"))
            if not start_loc or len(start_loc) < 2:
                continue
            start_x, start_y = float(start_loc[0]), float(start_loc[1])
            is_right_corner = 1 if start_y > 40.0 else 0

            # Determine Target Routine (Stage 1) using Euclidean pass distance and spatial corridor
            end_loc = parse_location(row.get("pass_end_location"))
            if not end_loc or len(end_loc) < 2:
                continue
            end_x, end_y = float(end_loc[0]), float(end_loc[1])
            pass_dist = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            if pass_dist < 22.0:
                target_routine = 2  # Short Corner Routine (pass length < 22 yards to supporting teammate)
            elif 30.0 <= end_y <= 50.0:
                target_routine = 0  # Direct Cross to Central Box / Penalty Spot corridor
            else:
                target_routine = 1  # Direct Cross to Near / Far Posts or Wide Box

            # Determine Target Outcome (Stage 2: Attacking Success vs Defensive Success)
            target_outcome = 0
            pass_outcome = str(row.get("pass_outcome", ""))
            if pass_outcome in ["Incomplete", "Out", "Pass Offside", "Unknown"]:
                target_outcome = 0
            elif row.get("pass_shot_assist") == True or row.get("pass_goal_assist") == True:
                target_outcome = 1
            else:
                # Check subsequent 15 events in same sequence
                sub_df = df.loc[c_idx + 1 : c_idx + 16]
                for _, s_row in sub_df.iterrows():
                    if str(s_row.get("play_pattern", "")) != "From Corner":
                        break
                    if str(s_row.get("team", "")) == corner_team and str(s_row.get("type", "")) in ["Shot", "Goal"]:
                        target_outcome = 1
                        break
                    if str(s_row.get("team", "")) != corner_team and str(s_row.get("type", "")) in [
                        "Clearance",
                        "Goal Keeper",
                    ]:
                        target_outcome = 0
                        break

            # Extract features
            minute = float(row.get("minute", 0.0))
            time_ratio = min(1.0, minute / 90.0)
            score_diff = (home_goals - away_goals) if corner_team == home_team else (away_goals - home_goals)
            is_home = 1 if corner_team == home_team else 0
            inswinging = (
                1
                if (row.get("pass_inswinging") == True or str(row.get("pass_technique", "")) == "Inswinging")
                else 0
            )

            taker = str(row.get("player", ""))
            taker_acc = player_profiles.get(taker, {}).get("accuracy", 0.75)
            taker_kp = player_profiles.get(taker, {}).get("progressive_ratio", 0.20)

            mgr_name = TEAM_TO_MANAGER.get(corner_team, "")
            team_dir = mgr_profiles.get(mgr_name, {}).get("directness", 5.0)
            team_wid = mgr_profiles.get(mgr_name, {}).get("width", 5.0)

            opp_team = away_team if corner_team == home_team else home_team
            opp_gk_save = 0.70
            if opp_team:
                # Find goalie for opp team
                for p_name, p_data in gk_profiles.items():
                    if p_data.get("team") == opp_team:
                        opp_gk_save = p_data.get("save_ratio", 0.70)
                        break
            opp_def_rate = def_profiles.get((str(opp_team), "Z_5_2"), 0.15)
            under_press = 1 if row.get("under_pressure") == True else 0

            # 1. Tactical & Routine History
            prev_routine = team_prev_routine.get(corner_team, -1)
            current_seconds = minute * 60.0 + float(row.get("second", 0.0))
            times_list = team_corner_times.get(corner_team, [])
            cluster_count = sum(1 for t in times_list if (current_seconds - t) <= 180.0)
            team_corner_times.setdefault(corner_team, []).append(current_seconds)

            # 2. Physical & Aerial Mismatches
            t_total = team_aerial_total.get(corner_team, 0)
            t_rate = team_aerial_wins.get(corner_team, 0) / t_total if t_total > 0 else 0.50
            o_total = team_aerial_total.get(opp_team, 0) if opp_team else 0
            o_rate = team_aerial_wins.get(opp_team, 0) / o_total if o_total > 0 else 0.50
            aerial_adv = float(t_rate - o_rate)

            gk_line_cmd = 0.70
            if (opp_gk_claims + opp_gk_punches) > 0:
                gk_line_cmd = float(opp_gk_claims / (opp_gk_claims + opp_gk_punches))

            # 3. Specialist Delivery Metrics
            t_count = taker_corners.get(taker, 0)
            t_ast = taker_assists.get(taker, 0)
            taker_assist_rate = float(t_ast / t_count) if t_count > 0 else 0.12

            # 4. Short-Term Lag Vector (last 5 routines across past games)
            past_routines = team_hist_routines.get(corner_team, [])
            lags = (past_routines[-5:] if len(past_routines) >= 5 else [ -1 ] * (5 - len(past_routines)) + past_routines)
            recent_5 = list(reversed(lags))
            r_lag_1, r_lag_2, r_lag_3, r_lag_4, r_lag_5 = recent_5[0], recent_5[1], recent_5[2], recent_5[3], recent_5[4]

            # 5. Long-Term Rolling Rates over last 20 corners across past games
            last_20 = past_routines[-20:]
            n_20 = len(last_20)
            rate_0 = float(last_20.count(0) / n_20) if n_20 > 0 else 0.75
            rate_2 = float(last_20.count(2) / n_20) if n_20 > 0 else 0.10

            # 6. Engineered Sequence Indicators
            consec_same = 0
            if r_lag_1 != -1:
                for r in reversed(past_routines):
                    if r == r_lag_1:
                        consec_same += 1
                    else:
                        break

            dataset.append(
                {
                    "is_right_corner": is_right_corner,
                    "time_ratio": time_ratio,
                    "score_differential": score_diff,
                    "is_home_team": is_home,
                    "inswinging": inswinging,
                    "taker_accuracy": taker_acc,
                    "taker_key_pass_ratio": taker_kp,
                    "team_directness": team_dir,
                    "team_width": team_wid,
                    "opp_gk_save_ratio": opp_gk_save,
                    "opp_def_rate": opp_def_rate,
                    "under_pressure": under_press,
                    "corner_cluster_density": cluster_count,
                    "aerial_height_advantage": aerial_adv,
                    "goalkeeper_line_command": gk_line_cmd,
                    "taker_corner_assist_rate": taker_assist_rate,
                    "routine_lag_1": r_lag_1,
                    "routine_lag_2": r_lag_2,
                    "routine_lag_3": r_lag_3,
                    "routine_lag_4": r_lag_4,
                    "routine_lag_5": r_lag_5,
                    "hist_rate_routine_0": rate_0,
                    "hist_rate_routine_2": rate_2,
                    "consecutive_same_routine": consec_same,
                    "target_routine": target_routine,
                    "target_outcome": target_outcome,
                }
            )

            # Update running stats
            team_prev_routine[corner_team] = target_routine
            team_hist_routines.setdefault(corner_team, []).append(target_routine)
            taker_corners[taker] = t_count + 1
            if target_outcome == 1:
                taker_assists[taker] = t_ast + 1

            # Update goal tracking if event was a goal
            if str(row.get("type")) == "Shot" and str(row.get("shot_outcome")) == "Goal":
                if corner_team == home_team:
                    home_goals += 1
                else:
                    away_goals += 1

        if mode == "iteration" and len(dataset) >= 600:
            print(f"  Reached {len(dataset)} records for iteration mode. Stopping extraction.")
            break

    df_out = pd.DataFrame(dataset)
    os.makedirs("data", exist_ok=True)
    out_path = f"data/corners_training_data_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Extracted {len(df_out)} corner kick records. Saved to {out_path}.")
    return df_out


@click.command()
@click.option("--mode", type=click.Choice(["iteration", "production"]), default="iteration", help="Extraction mode")
def main(mode: str) -> None:
    extract_corner_features(mode)


if __name__ == "__main__":
    main()
