import os
import pandas as pd
import numpy as np
from statsbombpy import sb
from utils import parse_location, map_coordinates_to_zone



def calculate_time_weight(match_date_str, ref_date, decay_lambda):
    """Calculates the exponential decay weight based on how recent the match was."""
    try:
        match_dt = pd.to_datetime(match_date_str)
        days_ago = (ref_date - match_dt).days
        # Ensure we don't have negative days if a match is played in the future relative to ref_date
        days_ago = max(0, days_ago)
        return np.exp(-decay_lambda * days_ago)
    except:
        return 1.0 # Default weight if date parsing fails

def build_self_contained_pipeline(max_matches_per_comp=30, ref_date="2026-06-30", decay_lambda=0.0019):
    print("Starting Recency-Weighted StatsBomb Data Pipeline...")
    ref_date_dt = pd.to_datetime(ref_date)
    competitions = sb.competitions()
    
    target_comp_names = [
        "FIFA World Cup", 
        "UEFA Euro", 
        "Copa America",
        "La Liga", 
        "Champions League",
        "Premier League",
        "1. Bundesliga",
        "Ligue 1"
    ]
    
    selected_competitions = competitions[
        competitions['competition_name'].isin(target_comp_names)
    ]
    
    print(f"Found {len(selected_competitions)} competition-seasons matching targets.")
    
    # Storage for weighted stats
    all_passes = [] # stores tuples: (start_zone, end_zone, event_type, weight)
    
    player_pass_counts = {} # player -> {completions, attempts, progressive} (all weighted)
    player_shot_counts = {} # player -> {shots, goals} (all weighted)
    goalkeeper_counts = {}   # goalkeeper -> {shots_faced, goals_conceded} (all weighted)
    
    team_defensive_actions = {} # team -> {zone -> weighted_count}
    team_weight_sums = {}       # team -> sum of match weights (for normalization)
    
    manager_stats = {} # manager -> {weighted_directness, weighted_width, weighted_tempo, weight_sum}
    
    match_count = 0
    for _, comp in selected_competitions.iterrows():
        comp_id = comp['competition_id']
        season_id = comp['season_id']
        
        print(f"\nFetching matches for {comp['competition_name']} ({comp['season_name']})...")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            match_slice = matches.copy()
            if max_matches_per_comp is not None:
                match_slice = match_slice.head(max_matches_per_comp)
                
            for _, match_row in match_slice.iterrows():
                match_id = match_row['match_id']
                match_date = match_row['match_date']
                
                # Get managers (StatsBomb uses plural column names)
                home_manager = match_row.get('home_managers')
                away_manager = match_row.get('away_managers')
                
                # Calculate time-decay weight
                weight = calculate_time_weight(match_date, ref_date_dt, decay_lambda)
                
                # Local Caching Logic: Load from disk if available, otherwise download and save.
                cache_dir = "data/raw_events"
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{match_id}.csv")
                
                try:
                    if os.path.exists(cache_path):
                        match_events = pd.read_csv(cache_path)
                    else:
                        match_events = sb.events(match_id=match_id)
                        match_events.to_csv(cache_path, index=False)
                except Exception as e:
                    print(f"  Error loading/caching match {match_id}: {e}")
                    continue
                
                # Track match weights for teams
                home_team = match_row['home_team']
                away_team = match_row['away_team']
                for team in [home_team, away_team]:
                    if team not in team_weight_sums:
                        team_weight_sums[team] = 0.0
                    team_weight_sums[team] += weight
                
                # --- 1. PASSES & PROGRESSION ---
                passes = match_events[match_events['type'] == 'Pass'].copy()
                match_total_passes = len(passes)
                match_progressive_passes = 0
                match_wing_passes = 0
                
                if not passes.empty:
                    passes = passes.dropna(subset=['location', 'pass_end_location', 'player'])
                    
                    # Parse locations if they are strings (CSV cache)
                    passes['location'] = passes['location'].apply(parse_location)
                    passes['pass_end_location'] = passes['pass_end_location'].apply(parse_location)
                    passes = passes.dropna(subset=['location', 'pass_end_location'])
                    
                    passes['start_zone'] = passes['location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
                    passes['end_zone'] = passes['pass_end_location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
                    passes['event_type'] = passes['pass_outcome'].apply(lambda outcome: 'Turnover' if pd.notnull(outcome) else 'Pass')
                    
                    # Store passes with weights for global baseline matrix
                    for _, row in passes.iterrows():
                        all_passes.append({
                            'start_zone': row['start_zone'],
                            'end_zone': row['end_zone'],
                            'event_type': row['event_type'],
                            'weight': weight
                        })
                        
                        player = row['player']
                        is_successful = row['event_type'] == 'Pass'
                        start_x = int(row['start_zone'].split('_')[1])
                        end_x = int(row['end_zone'].split('_')[1])
                        end_y = int(row['end_zone'].split('_')[2])
                        
                        is_progressive = is_successful and (end_x > start_x)
                        is_wing = is_successful and (end_y in [0, 4])
                        
                        if is_progressive:
                            match_progressive_passes += 1
                        if is_wing:
                            match_wing_passes += 1
                            
                        # Accumulate weighted player passing
                        if player not in player_pass_counts:
                            player_pass_counts[player] = {'completions': 0.0, 'attempts': 0.0, 'progressive': 0.0}
                        player_pass_counts[player]['attempts'] += weight
                        if is_successful:
                            player_pass_counts[player]['completions'] += weight
                            if is_progressive:
                                player_pass_counts[player]['progressive'] += weight
                
                # --- 2. SHOTS & GOALS (FOR SHOOTERS) ---
                shots = match_events[match_events['type'] == 'Shot'].copy()
                if not shots.empty:
                    shots = shots.dropna(subset=['player'])
                    for _, row in shots.iterrows():
                        player = row['player']
                        is_goal = row['shot_outcome'] == 'Goal'
                        
                        if player not in player_shot_counts:
                            player_shot_counts[player] = {'shots': 0.0, 'goals': 0.0}
                        player_shot_counts[player]['shots'] += weight
                        if is_goal:
                            player_shot_counts[player]['goals'] += weight
                
                # --- 3. GOALKEEPERS ---
                gk_events = match_events[match_events['type'] == 'Goal Keeper'].copy()
                if not gk_events.empty:
                    shot_facings = gk_events[gk_events['goalkeeper_type'].isin(['Shot Faced', 'Penalty Faced'])]
                    shot_facings = shot_facings.dropna(subset=['player'])
                    for _, row in shot_facings.iterrows():
                        gk = row['player']
                        is_conceded = row['goalkeeper_outcome'] in ['Goal Conceded', 'Incomplete']
                        
                        if gk not in goalkeeper_counts:
                            goalkeeper_counts[gk] = {'shots_faced': 0.0, 'goals_conceded': 0.0}
                        goalkeeper_counts[gk]['shots_faced'] += weight
                        if is_conceded:
                            goalkeeper_counts[gk]['goals_conceded'] += weight
                
                # --- 4. TEAM DEFENSIVE ACTIONS BY ZONE ---
                defensive_types = ['Duel', 'Interception', 'Clearance', 'Block', 'Foul Committed', 'Pressure']
                def_events = match_events[match_events['type'].isin(defensive_types)].copy()
                if not def_events.empty:
                    def_events = def_events.dropna(subset=['location', 'team'])
                    def_events['location'] = def_events['location'].apply(parse_location)
                    def_events = def_events.dropna(subset=['location'])
                    
                    def_events['zone'] = def_events['location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
                    def_events = def_events.dropna(subset=['zone'])
                    
                    for _, row in def_events.iterrows():
                        team = row['team']
                        zone = row['zone']
                        
                        if team not in team_defensive_actions:
                            team_defensive_actions[team] = {}
                        if zone not in team_defensive_actions[team]:
                            team_defensive_actions[team][zone] = 0.0
                        team_defensive_actions[team][zone] += weight
                        
                # --- 5. DYNAMIC MANAGER TACTICS ---
                # Calculate match-level tactical metrics
                if match_total_passes > 10:
                    match_directness = match_progressive_passes / match_total_passes
                    match_width = match_wing_passes / match_total_passes
                    
                    # Calculate tempo (possession chains)
                    # We can estimate tempo by counting the number of possession changes
                    possession_changes = match_events['possession'].diff().fillna(0).ne(0).sum()
                    match_tempo = possession_changes
                    
                    # Accumulate for both managers
                    for manager in [home_manager, away_manager]:
                        if pd.notnull(manager):
                            if manager not in manager_stats:
                                manager_stats[manager] = {
                                    'directness_sum': 0.0, 
                                    'width_sum': 0.0, 
                                    'tempo_sum': 0.0, 
                                    'weight_sum': 0.0
                                }
                            manager_stats[manager]['directness_sum'] += match_directness * weight
                            manager_stats[manager]['width_sum'] += match_width * weight
                            manager_stats[manager]['tempo_sum'] += match_tempo * weight
                            manager_stats[manager]['weight_sum'] += weight
                
                match_count += 1
                if match_count % 10 == 0:
                    print(f"  Processed {match_count} matches...")
                    
        except Exception as e:
            print(f"Skipping comp_id {comp_id} due to error: {e}")
            
    if not all_passes:
        print("No pass data was successfully downloaded.")
        return
        
    os.makedirs("data", exist_ok=True)
    
    # --- SAVE 1. Global Baseline Matrix (Weighted) ---
    print("\nCompiling Global Baseline Matrix...")
    df_global_passes = pd.DataFrame(all_passes)
    zones = [f"Z_{x}_{y}" for x in range(6) for y in range(5)]
    matrix = pd.DataFrame(0.0, index=zones, columns=zones)
    
    # Group by start and end zone, summing the weights
    successful_passes = df_global_passes[df_global_passes['event_type'] == 'Pass']
    transitions = successful_passes.groupby(['start_zone', 'end_zone'])['weight'].sum().reset_index(name='weighted_count')
    
    for _, row in transitions.iterrows():
        matrix.at[row['start_zone'], row['end_zone']] = row['weighted_count']
        
    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
    matrix.to_csv("data/global_baseline_matrix.csv")
    print("Saved global baseline matrix.")
    
    # --- SAVE 2. Player Profiles (Passing & Shooting, Weighted) ---
    player_records = []
    all_players = set(player_pass_counts.keys()).union(set(player_shot_counts.keys()))
    for player in all_players:
        pass_stats = player_pass_counts.get(player, {'completions': 0.0, 'attempts': 0.0, 'progressive': 0.0})
        shot_stats = player_shot_counts.get(player, {'shots': 0.0, 'goals': 0.0})
        
        attempts = pass_stats['attempts']
        completions = pass_stats['completions']
        progressive = pass_stats['progressive']
        
        shots = shot_stats['shots']
        goals = shot_stats['goals']
        
        accuracy = completions / attempts if attempts > 0 else 0.0
        progressive_ratio = progressive / completions if completions > 0 else 0.0
        shot_conversion = goals / shots if shots > 0 else 0.0
        
        player_records.append({
            'player': player,
            'attempts': attempts, # Weighted attempts (acting as effective sample size)
            'completions': completions,
            'progressive': progressive,
            'accuracy': accuracy,
            'progressive_ratio': progressive_ratio,
            'shots': shots,
            'goals': goals,
            'shot_conversion': shot_conversion
        })
        
    df_player_profiles = pd.DataFrame(player_records)
    # Filter out players with low effective sample size (weighted attempts < 5.0)
    df_player_profiles = df_player_profiles[(df_player_profiles['attempts'] >= 5.0) | (df_player_profiles['shots'] >= 1.0)]
    df_player_profiles.to_csv("data/statsbomb_player_profiles.csv", index=False)
    print("Saved player profiles.")
    
    # --- SAVE 3. Goalkeeper Profiles (Weighted) ---
    gk_records = []
    for gk, stats in goalkeeper_counts.items():
        faced = stats['shots_faced']
        conceded = stats['goals_conceded']
        saves = faced - conceded
        save_ratio = saves / faced if faced > 0 else 0.70
        
        gk_records.append({
            'goalkeeper': gk,
            'shots_faced': faced,
            'goals_conceded': conceded,
            'save_ratio': save_ratio
        })
    df_gk = pd.DataFrame(gk_records)
    df_gk.to_csv("data/goalkeeper_profiles.csv", index=False)
    print("Saved goalkeeper profiles.")
    
    # --- SAVE 4. Team Defensive Profiles (Normalized by sum of match weights) ---
    def_records = []
    for team, zones_dict in team_defensive_actions.items():
        total_weight = team_weight_sums.get(team, 1.0)
        for zone, count in zones_dict.items():
            rate = count / total_weight # Weighted defensive rate per match
            def_records.append({
                'team': team,
                'zone': zone,
                'defensive_rate': rate
            })
    df_def = pd.DataFrame(def_records)
    df_def.to_csv("data/team_defensive_profiles.csv", index=False)
    print("Saved team defensive profiles.")
    
    # --- SAVE 5. Dynamic Manager Profiles (Weighted & Normalized) ---
    mgr_records = []
    for manager, stats in manager_stats.items():
        w_sum = stats['weight_sum']
        if w_sum > 0:
            # Normalize to get the weighted average
            avg_directness = stats['directness_sum'] / w_sum
            avg_width = stats['width_sum'] / w_sum
            avg_tempo = stats['tempo_sum'] / w_sum
            
            # Map these raw ratios to a 1-10 scale for the MCMC engine
            # e.g., Directness: average is ~0.3. Map 0.1 -> 1, 0.3 -> 5, 0.5 -> 10
            directness_score = max(1, min(10, int(avg_directness * 20.0)))
            width_score = max(1, min(10, int(avg_width * 20.0)))
            
            # Tempo: average is ~180 possession changes. Map 120 -> 2, 180 -> 5, 240 -> 8
            tempo_score = max(1, min(10, int((avg_tempo - 120) / 20.0) + 2))
            
            mgr_records.append({
                'manager': manager,
                'directness': directness_score,
                'width': width_score,
                'tempo': tempo_score,
                'raw_directness': avg_directness,
                'raw_width': avg_width,
                'raw_tempo': avg_tempo
            })
    df_mgr = pd.DataFrame(mgr_records)
    df_mgr.to_csv("data/manager_profiles.csv", index=False)
    print("Saved dynamic manager profiles.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download StatsBomb data and compile profiles.")
    parser.add_argument('--max-matches', type=int, default=30, help="Max matches per competition-season (None for all)")
    parser.add_argument('--ref-date', type=str, default="2026-06-30", help="Reference date for time decay")
    parser.add_argument('--decay-lambda', type=float, default=0.0019, help="Lambda parameter for exponential decay")
    args = parser.parse_args()
    
    build_self_contained_pipeline(
        max_matches_per_comp=args.max_matches,
        ref_date=args.ref_date,
        decay_lambda=args.decay_lambda
    )
