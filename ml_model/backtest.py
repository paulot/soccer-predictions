import os
import pandas as pd
import click
from statsbombpy import sb
from mcmc_simulation import build_30_zone_grid, map_coordinates_to_zone
from ml_model.simulator import simulate_full_match
from utils import parse_location, TEAM_TO_MANAGER, calculate_brier_score, calculate_log_loss
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any, Tuple, Optional


def run_single_simulation(args: Tuple[Any, ...]) -> Tuple[int, int]:
    (
        model_type,
        outcome_path,
        dest_path,
        base_matrix,
        home_team,
        away_team,
        df_target_passes,
        player_profiles,
        gk_profiles,
        team_defensive_profiles,
        manager_profiles,
        TEAM_TO_MANAGER_dict,
        player_to_team,
        zones,
        dynamic_possessions,
    ) = args

    # Import inside worker to prevent circular import issues
    from ml_model.models import HeuristicTransitionModel, MLTransitionModel

    if model_type == "heuristic":
        model: Any = HeuristicTransitionModel(base_matrix)
    else:
        model = MLTransitionModel(outcome_path, dest_path)

    return simulate_full_match(
        home_team,
        away_team,
        model,
        df_target_passes,
        player_profiles,
        gk_profiles,
        team_defensive_profiles,
        manager_profiles,
        TEAM_TO_MANAGER_dict,
        player_to_team,
        zones,
        num_possessions=dynamic_possessions,
    )


def run_ml_backtest(
    model_type: str = "random_forest", mode: str = "iteration", num_simulations: int = 500
) -> Optional[Dict[str, float]]:
    print("\n==================================================")
    print(f"STARTING BACKTEST: Model = {model_type.upper()} | Mode = {mode.upper()}")
    print("==================================================")

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
            t = str(row["team"])
            z = str(row["zone"])
            r = float(row["defensive_rate"])
            if t not in team_defensive_profiles:
                team_defensive_profiles[t] = {}
            team_defensive_profiles[t][z] = r

        df_mgr: pd.DataFrame = pd.read_csv("data/manager_profiles.csv")
        manager_profiles: Dict[str, Dict[str, float]] = df_mgr.set_index("manager").to_dict(orient="index")
    except Exception as e:
        print(f"Error loading global datasets: {e}. Please run download_data.py first.")
        return None

    # 2. Instantiate the Modular Transition Model (only check if path exists if not heuristic)
    if model_type != "heuristic":
        outcome_path: str = f"data/models/{model_type}_{mode}_outcome.pkl"
        dest_path: str = f"data/models/{model_type}_{mode}_destination.pkl"

        if not os.path.exists(outcome_path) or not os.path.exists(dest_path):
            print(f"Models not found at {outcome_path}. Training them now...")
            from ml_model.train import train_models

            train_models(model_type, mode)

    # 3. Query Target Match IDs (World Cup 2022)
    competitions: pd.DataFrame = sb.competitions()
    wc_2022 = competitions[
        (competitions["competition_name"] == "FIFA World Cup") & (competitions["season_name"] == "2022")
    ].iloc[0]
    matches: pd.DataFrame = sb.matches(competition_id=wc_2022["competition_id"], season_id=wc_2022["season_id"])

    # Mode-specific backtest configurations
    match_ids: List[int] = []
    if mode == "iteration":
        match_ids = matches["match_id"].head(2).tolist()
        num_simulations = min(100, num_simulations)
        print(f"  Iteration Mode: Limiting to {len(match_ids)} matches with {num_simulations} simulations each.")
    else:
        match_ids = matches["match_id"].head(5).tolist()
        print(f"  Production Mode: Evaluating {len(match_ids)} matches with {num_simulations} simulations each.")

    zones: List[str] = build_30_zone_grid()
    results: List[Dict[str, Any]] = []

    # 4. Pre-load match events for evaluation
    all_events: Dict[int, pd.DataFrame] = {}
    match_details: Dict[int, Dict[str, Any]] = {}
    for mid in match_ids:
        try:
            cache_path: str = f"data/raw_events/{mid}.csv"
            if os.path.exists(cache_path):
                events = pd.read_csv(cache_path)
            else:
                events = sb.events(match_id=mid)
            all_events[mid] = events

            home_team: str = str(events["team"].dropna().unique()[0])
            away_team: str = str(events["team"].dropna().unique()[1])

            shots = events[events["type"] == "Shot"]
            home_actual: int = len(shots[(shots["team"] == home_team) & (shots["shot_outcome"] == "Goal")])
            away_actual: int = len(shots[(shots["team"] == away_team) & (shots["shot_outcome"] == "Goal")])

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

    # 5. Run the Backtest Loop
    for target_mid in list(all_events.keys()):
        target_events = all_events[target_mid]
        home_team = match_details[target_mid]["home_team"]
        away_team = match_details[target_mid]["away_team"]

        print(f"\nEvaluating Match {target_mid}: {home_team} vs {away_team}")

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

        player_to_team: Dict[str, str] = (
            target_events.dropna(subset=["player", "team"]).set_index("player")["team"].to_dict()
        )

        # Calculate dynamic tempo based on manager profiles (baseline 100 possessions)
        home_mgr = manager_profiles.get(TEAM_TO_MANAGER.get(home_team, ""), {"tempo": 5.0})
        away_mgr = manager_profiles.get(TEAM_TO_MANAGER.get(away_team, ""), {"tempo": 5.0})
        tempo_factor: float = 1.0 + (home_mgr["tempo"] + away_mgr["tempo"] - 10) * 0.05
        dynamic_possessions: int = int(100 * tempo_factor)

        home_wins: int = 0
        draws: int = 0
        away_wins: int = 0

        print(f"Simulating match {num_simulations} times ({dynamic_possessions} possessions/game) in parallel...")

        outcome_path_sim: str = f"data/models/{model_type}_{mode}_outcome.pkl" if model_type != "heuristic" else ""
        dest_path_sim: str = f"data/models/{model_type}_{mode}_destination.pkl" if model_type != "heuristic" else ""

        sim_args = (
            model_type,
            outcome_path_sim,
            dest_path_sim,
            base_matrix,
            home_team,
            away_team,
            df_target_passes,
            player_profiles,
            gk_profiles,
            team_defensive_profiles,
            manager_profiles,
            TEAM_TO_MANAGER,
            player_to_team,
            zones,
            dynamic_possessions,
        )

        with ProcessPoolExecutor() as executor:
            sim_results = list(executor.map(run_single_simulation, [sim_args] * num_simulations))

        for h_goals, a_goals in sim_results:
            if h_goals > a_goals:
                home_wins += 1
            elif h_goals == a_goals:
                draws += 1
            else:
                away_wins += 1

        prob_win: float = home_wins / num_simulations
        prob_draw: float = draws / num_simulations
        prob_loss: float = away_wins / num_simulations

        actual_outcome: str = match_details[target_mid]["actual_outcome"]
        brier: float = calculate_brier_score(prob_win, prob_draw, prob_loss, actual_outcome)
        logloss: float = calculate_log_loss(prob_win, prob_draw, prob_loss, actual_outcome)

        predicted_outcome: str = (
            "W" if prob_win > prob_draw and prob_win > prob_loss else ("D" if prob_draw > prob_loss else "L")
        )
        is_correct: int = 1 if predicted_outcome == actual_outcome else 0

        print(f"Predictions: Win={prob_win:.2%}, Draw={prob_draw:.2%}, Loss={prob_loss:.2%}")
        print(
            f"Actual Outcome: {actual_outcome} (Score: {match_details[target_mid]['home_actual']}-{match_details[target_mid]['away_actual']})"  # noqa: E501
        )
        print(f"Brier Score: {brier:.4f} | Log Loss: {logloss:.4f} | Correct: {is_correct}")

        results.append({"match_id": target_mid, "brier": brier, "log_loss": logloss, "correct": is_correct})

    df_results: pd.DataFrame = pd.DataFrame(results)
    print("\n==================================================")
    print(f"SUMMARY RESULTS: Model = {model_type.upper()}")
    print("==================================================")
    print(f"Average Brier Score: {df_results['brier'].mean():.4f}")
    print(f"Average Log Loss:    {df_results['log_loss'].mean():.4f}")
    print(f"Overall Accuracy:    {df_results['correct'].mean():.2%}")
    print("==================================================")

    return {
        "brier": float(df_results["brier"].mean()),
        "log_loss": float(df_results["log_loss"].mean()),
        "accuracy": float(df_results["correct"].mean()),
    }


@click.command(help="Run MCMC backtest for a specific model.")
@click.option(
    "--model",
    type=click.Choice(["heuristic", "logistic_regression", "random_forest", "xgboost", "neural_network"]),
    default="random_forest",
    help="Model to evaluate",
)
@click.option(
    "--mode", type=click.Choice(["iteration", "production"]), default="iteration", help="Mode: iteration or production"
)
@click.option("--sims", type=int, default=500, help="Number of MCMC simulations per match")
def main(model: str, mode: str, sims: int) -> None:
    run_ml_backtest(model, mode, sims)


if __name__ == "__main__":
    main()
