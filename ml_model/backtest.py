import os
import pandas as pd
import numpy as np
import argparse
from statsbombpy import sb
from mcmc_simulation import build_30_zone_grid, map_coordinates_to_zone
from ml_model.models import HeuristicTransitionModel, MLTransitionModel
from ml_model.simulator import simulate_full_match
from download_data import parse_location

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

def run_ml_backtest(model_type='random_forest', mode='iteration', num_simulations=500):
    print(f"\n==================================================")
    print(f"STARTING BACKTEST: Model = {model_type.upper()} | Mode = {mode.upper()}")
    print(f"==================================================")
    
    # 1. Load Global Datasets
    try:
        base_matrix = pd.read_csv("data/global_baseline_matrix.csv", index_col=0)
        df_profiles = pd.read_csv("data/statsbomb_player_profiles.csv")
        player_profiles = df_profiles.set_index('player').to_dict(orient='index')
        
        df_gk = pd.read_csv("data/goalkeeper_profiles.csv")
        gk_profiles = df_gk.set_index('goalkeeper')['save_ratio'].to_dict()
        
        df_def = pd.read_csv("data/team_defensive_profiles.csv")
        team_defensive_profiles = {}
        for _, row in df_def.iterrows():
            t = row['team']
            z = row['zone']
            r = row['defensive_rate']
            if t not in team_defensive_profiles:
                team_defensive_profiles[t] = {}
            team_defensive_profiles[t][z] = r
            
        df_mgr = pd.read_csv("data/manager_profiles.csv")
        manager_profiles = df_mgr.set_index('manager').to_dict(orient='index')
    except Exception as e:
        print(f"Error loading global datasets: {e}. Please run download_data.py first.")
        return
        
    # 2. Instantiate the Modular Transition Model
    if model_type == 'heuristic':
        model = HeuristicTransitionModel(base_matrix)
    else:
        outcome_path = f"data/models/{model_type}_{mode}_outcome.pkl"
        dest_path = f"data/models/{model_type}_{mode}_destination.pkl"
        
        if not os.path.exists(outcome_path) or not os.path.exists(dest_path):
            print(f"Models not found at {outcome_path}. Training them now...")
            from ml_model.train import train_models
            train_models(model_type, mode)
            
        model = MLTransitionModel(outcome_path, dest_path)
        
    # 3. Query Target Match IDs (World Cup 2022)
    competitions = sb.competitions()
    wc_2022 = competitions[
        (competitions['competition_name'] == "FIFA World Cup") & 
        (competitions['season_name'] == "2022")
    ].iloc[0]
    matches = sb.matches(competition_id=wc_2022['competition_id'], season_id=wc_2022['season_id'])
    
    # Mode-specific backtest configurations
    if mode == 'iteration':
        match_ids = matches['match_id'].head(2).tolist()
        num_simulations = min(100, num_simulations)
        print(f"  Iteration Mode: Limiting to {len(match_ids)} matches with {num_simulations} simulations each.")
    else:
        match_ids = matches['match_id'].head(5).tolist()
        print(f"  Production Mode: Evaluating {len(match_ids)} matches with {num_simulations} simulations each.")
    
    zones = build_30_zone_grid()
    results = []
    
    # 4. Pre-load match events for evaluation
    all_events = {}
    match_details = {}
    for mid in match_ids:
        try:
            cache_path = f"data/raw_events/{mid}.csv"
            if os.path.exists(cache_path):
                events = pd.read_csv(cache_path)
            else:
                events = sb.events(match_id=mid)
            all_events[mid] = events
            
            home_team = events['team'].dropna().unique()[0]
            away_team = events['team'].dropna().unique()[1]
            
            shots = events[events['type'] == 'Shot']
            home_actual = len(shots[(shots['team'] == home_team) & (shots['shot_outcome'] == 'Goal')])
            away_actual = len(shots[(shots['team'] == away_team) & (shots['shot_outcome'] == 'Goal')])
            
            if home_actual > away_actual:
                actual_outcome = 'W'
            elif home_actual == away_actual:
                actual_outcome = 'D'
            else:
                actual_outcome = 'L'
                
            match_details[mid] = {
                'home_team': home_team,
                'away_team': away_team,
                'home_actual': home_actual,
                'away_actual': away_actual,
                'actual_outcome': actual_outcome
            }
        except Exception as e:
            print(f"Skipping match {mid} due to load error: {e}")
            
    # 5. Run the Backtest Loop
    for target_mid in list(all_events.keys()):
        target_events = all_events[target_mid]
        home_team = match_details[target_mid]['home_team']
        away_team = match_details[target_mid]['away_team']
        
        print(f"\nEvaluating Match {target_mid}: {home_team} vs {away_team}")
        
        df_target_passes = target_events[target_events['type'] == 'Pass'].copy()
        df_target_passes = df_target_passes.dropna(subset=['location', 'pass_end_location', 'player'])
        
        # Parse locations (handles string format from CSV cache)
        df_target_passes['location'] = df_target_passes['location'].apply(parse_location)
        df_target_passes['pass_end_location'] = df_target_passes['pass_end_location'].apply(parse_location)
        df_target_passes = df_target_passes.dropna(subset=['location', 'pass_end_location'])
        
        df_target_passes['start_zone'] = df_target_passes['location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
        df_target_passes['end_zone'] = df_target_passes['pass_end_location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
        
        player_to_team = target_events.dropna(subset=['player', 'team']).set_index('player')['team'].to_dict()
        
        # Calculate dynamic tempo based on manager profiles (baseline 100 possessions)
        home_mgr = manager_profiles.get(TEAM_TO_MANAGER.get(home_team), {"tempo": 5})
        away_mgr = manager_profiles.get(TEAM_TO_MANAGER.get(away_team), {"tempo": 5})
        tempo_factor = 1.0 + (home_mgr['tempo'] + away_mgr['tempo'] - 10) * 0.05
        dynamic_possessions = int(100 * tempo_factor)
        
        home_wins = 0
        draws = 0
        away_wins = 0
        
        print(f"Simulating match {num_simulations} times ({dynamic_possessions} possessions/game)...")
        for _ in range(num_simulations):
            h_goals, a_goals = simulate_full_match(
                home_team, away_team, model, df_target_passes, 
                player_profiles, gk_profiles, team_defensive_profiles, 
                manager_profiles, TEAM_TO_MANAGER, player_to_team, zones,
                num_possessions=dynamic_possessions
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
        
        actual_outcome = match_details[target_mid]['actual_outcome']
        brier = calculate_brier_score(prob_win, prob_draw, prob_loss, actual_outcome)
        logloss = calculate_log_loss(prob_win, prob_draw, prob_loss, actual_outcome)
        
        predicted_outcome = 'W' if prob_win > prob_draw and prob_win > prob_loss else ('D' if prob_draw > prob_loss else 'L')
        is_correct = 1 if predicted_outcome == actual_outcome else 0
        
        print(f"Predictions: Win={prob_win:.2%}, Draw={prob_draw:.2%}, Loss={prob_loss:.2%}")
        print(f"Actual Outcome: {actual_outcome} (Score: {match_details[target_mid]['home_actual']}-{match_details[target_mid]['away_actual']})")
        print(f"Brier Score: {brier:.4f} | Log Loss: {logloss:.4f} | Correct: {is_correct}")
        
        results.append({
            'match_id': target_mid,
            'brier': brier,
            'log_loss': logloss,
            'correct': is_correct
        })
        
    df_results = pd.DataFrame(results)
    print("\n==================================================")
    print(f"SUMMARY RESULTS: Model = {model_type.upper()}")
    print("==================================================")
    print(f"Average Brier Score: {df_results['brier'].mean():.4f}")
    print(f"Average Log Loss:    {df_results['log_loss'].mean():.4f}")
    print(f"Overall Accuracy:    {df_results['correct'].mean():.2%}")
    print("==================================================")
    
    return {
        'brier': df_results['brier'].mean(),
        'log_loss': df_results['log_loss'].mean(),
        'accuracy': df_results['correct'].mean()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='random_forest', 
                        choices=['heuristic', 'logistic_regression', 'random_forest', 'xgboost'])
    parser.add_argument('--mode', type=str, default='iteration', choices=['iteration', 'production'])
    parser.add_argument('--sims', type=int, default=500)
    args = parser.parse_args()
    
    run_ml_backtest(args.model, args.mode, args.sims)
