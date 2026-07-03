import os
import ast
import pandas as pd
import numpy as np

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
    if pd.isnull(x) or pd.isnull(y):
        return None
    zone_x = min(int(x / 20), 5)
    zone_y = min(int(y / 16), 4)
    return f"Z_{zone_x}_{zone_y}"

def parse_timestamp_to_seconds(ts_str):
    if pd.isna(ts_str): return 0.0
    try:
        parts = ts_str.split(':')
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except:
        return 0.0

def extract_features_and_targets(mode='iteration'):
    print(f"Extracting features for ML Transition Model (Mode: {mode.upper()})...")
    
    import pickle
    
    # 1. Load profiles for joining
    try:
        df_players = pd.read_csv("data/statsbomb_player_profiles.csv")
        player_profiles = df_players.set_index('player').to_dict(orient='index')
        
        df_gk = pd.read_csv("data/goalkeeper_profiles.csv")
        gk_profiles = df_gk.set_index('goalkeeper').to_dict(orient='index')
        
        df_def = pd.read_csv("data/team_defensive_profiles.csv")
        # Map (team, zone) -> defensive_rate
        def_profiles = {}
        for _, row in df_def.iterrows():
            def_profiles[(row['team'], row['zone'])] = row['defensive_rate']
            
        df_mgr = pd.read_csv("data/manager_profiles.csv")
        mgr_profiles = df_mgr.set_index('manager').to_dict(orient='index')
    except Exception as e:
        print(f"Error loading profiles: {e}. Please run download_data.py first.")
        return

    # Load spectral embeddings
    emb_dir = "data/embeddings"
    if not os.path.exists(os.path.join(emb_dir, "zone_embeddings.pkl")):
        print("  Embeddings not found. Training spectral embeddings first...")
        from ml_model.train_embeddings import train_all_embeddings
        train_all_embeddings(mode)
        
    with open(os.path.join(emb_dir, "zone_embeddings.pkl"), 'rb') as f:
        zone_embeddings = pickle.load(f)
    with open(os.path.join(emb_dir, "player_embeddings.pkl"), 'rb') as f:
        player_embeddings = pickle.load(f)
    with open(os.path.join(emb_dir, "manager_embeddings.pkl"), 'rb') as f:
        manager_embeddings = pickle.load(f)

    # Mapping of teams to managers (for tactical lookup)
    from ml_model.backtest import TEAM_TO_MANAGER

    raw_dir = "data/raw_events"
    if not os.path.exists(raw_dir):
        print(f"Raw events directory {raw_dir} not found. Please run download_data.py first.")
        return
        
    match_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    # In iteration mode, only use a small subset of matches (e.g. 50)
    if mode == 'iteration':
        match_files = match_files[:50]
        print(f"  Iteration Mode: Limiting feature extraction to first {len(match_files)} matches.")
    
    dataset = []
    player_positions = {}
    
    ROLE_MAPPING = {
        'Goalkeeper': 0,
        'Right Back': 1, 'Left Back': 1, 'Center Back': 1, 'Right Center Back': 1, 'Left Center Back': 1,
        'Right Wing Back': 1, 'Left Wing Back': 1,
        'Center Defensive Midfield': 2, 'Right Center Midfield': 2, 'Left Center Midfield': 2,
        'Right Midfield': 2, 'Left Midfield': 2, 'Center Midfield': 2, 'Center Attacking Midfield': 2,
        'Right Wing': 3, 'Left Wing': 3, 'Center Forward': 3, 'Secondary Striker': 3,
        'Right Center Forward': 3, 'Left Center Forward': 3
    }
    
    for idx, filename in enumerate(match_files):
        match_path = os.path.join(raw_dir, filename)
        try:
            df_events = pd.read_csv(match_path)
        except:
            continue
            
        # Sort chronologically
        df_events = df_events.sort_values(by=['period', 'timestamp'])
        
        home_team = df_events['team'].dropna().unique()[0]
        away_team = df_events['team'].dropna().unique()[1]
        
        # Track running score
        home_score = 0
        away_score = 0
        
        # Track possession start times
        poss_start_times = {}
        for poss_id, gp in df_events.groupby('possession'):
            first_event = gp.iloc[0]
            ts = first_event.get('timestamp')
            poss_start_times[poss_id] = parse_timestamp_to_seconds(ts)
            
        # Accumulate player positions
        df_pos = df_events.dropna(subset=['player', 'position'])
        for _, row in df_pos.iterrows():
            p = row['player']
            pos = row['position']
            if p not in player_positions:
                player_positions[p] = {}
            player_positions[p][pos] = player_positions[p].get(pos, 0) + 1
            
        # Find all goalkeepers
        player_to_team = df_events.dropna(subset=['player', 'team']).set_index('player')['team'].to_dict()
        home_gk = None
        away_gk = None
        for player, team in player_to_team.items():
            if player in gk_profiles:
                if team == home_team:
                    home_gk = player
                elif team == away_team:
                    away_gk = player
                    
        # Extract passes with score and time context chronologically
        passes_with_context = []
        for _, row in df_events.iterrows():
            etype = row['type']
            team = row['team']
            
            # Update score
            if etype == 'Shot' and row.get('shot_outcome') == 'Goal':
                if team == home_team:
                    home_score += 1
                elif team == away_team:
                    away_score += 1
                    
            if etype == 'Pass':
                loc = parse_location(row['location'])
                end_loc = parse_location(row['pass_end_location'])
                if not loc or not end_loc or len(loc) < 2 or len(end_loc) < 2:
                    continue
                    
                pass_time = parse_timestamp_to_seconds(row.get('timestamp'))
                poss_id = row['possession']
                poss_start = poss_start_times.get(poss_id, pass_time)
                duration = max(0.0, pass_time - poss_start)
                
                score_diff = home_score - away_score if team == home_team else away_score - home_score
                
                passes_with_context.append({
                    'event_row': row.to_dict(),
                    'loc_parsed': loc,
                    'pass_end_loc_parsed': end_loc,
                    'score_differential': score_diff,
                    'possession_duration': duration
                })
                
        if not passes_with_context:
            continue
            
        # Group passes by possession to build the sequence history
        df_passes_context = pd.DataFrame(passes_with_context)
        for poss_id, group in df_passes_context.groupby(lambda idx: passes_with_context[idx]['event_row']['possession']):
            poss_passes = group.to_dict(orient='records')
            
            possession_directions = []
            
            for i, p in enumerate(poss_passes):
                row_dict = p['event_row']
                loc = p['loc_parsed']
                end_loc = p['pass_end_loc_parsed']
                
                curr_x = min(int(loc[0] / 20), 5)
                curr_y = min(int(loc[1] / 16), 4)
                end_x = min(int(end_loc[0] / 20), 5)
                end_y = min(int(end_loc[1] / 16), 4)
                
                player = row_dict['player']
                team = row_dict['team']
                opp_team = away_team if team == home_team else home_team
                opp_gk = away_gk if team == home_team else home_gk
                
                outcome = 1 if pd.notnull(row_dict.get('pass_outcome')) else 0
                
                p_profile = player_profiles.get(player, {'accuracy': 0.80, 'progressive_ratio': 0.25})
                passer_acc = p_profile['accuracy']
                passer_prog = p_profile['progressive_ratio']
                
                start_zone = f"Z_{curr_x}_{curr_y}"
                opp_def_rate = def_profiles.get((opp_team, start_zone), 0.0)
                opp_gk_profile = gk_profiles.get(opp_gk, {'save_ratio': 0.70})
                opp_gk_save = opp_gk_profile['save_ratio']
                
                mgr_name = TEAM_TO_MANAGER.get(team)
                m_profile = mgr_profiles.get(mgr_name, {'directness': 5, 'width': 5})
                mgr_dir = m_profile['directness']
                mgr_wid = m_profile['width']
                
                # Calculate spatial features based on discrete zone centers
                start_cx = curr_x * 20 + 10
                start_cy = curr_y * 16 + 8
                end_cx = end_x * 20 + 10
                end_cy = end_y * 16 + 8
                dx = end_cx - start_cx
                dy = end_cy - start_cy
                pass_length = np.sqrt(dx**2 + dy**2)
                pass_angle = np.arctan2(dy, dx)
                
                # Direction of this pass
                pass_dx = end_x - curr_x
                direction = 1 if pass_dx > 0 else (-1 if pass_dx < 0 else 0)
                
                # Extract history of last 3 passes in this possession
                prev_dir_1 = possession_directions[-1] if len(possession_directions) >= 1 else 0
                prev_dir_2 = possession_directions[-2] if len(possession_directions) >= 2 else 0
                prev_dir_3 = possession_directions[-3] if len(possession_directions) >= 3 else 0
                
                possession_directions.append(direction)
                
                # History features (N=2)
                history = {}
                for n in range(1, 3):
                    if i - n >= 0:
                        prev_p = poss_passes[i - n]
                        prev_loc = prev_p['loc_parsed']
                        if prev_loc and len(prev_loc) >= 2:
                            px = min(int(prev_loc[0] / 20), 5)
                            py = min(int(prev_loc[1] / 16), 4)
                        else:
                            px, py = -1, -1
                        p_outcome = 1 if pd.notnull(prev_p['event_row'].get('pass_outcome')) else 0
                        
                        history[f'prev_{n}_zone_x'] = px
                        history[f'prev_{n}_zone_y'] = py
                        history[f'prev_{n}_success'] = 1 - p_outcome
                    else:
                        history[f'prev_{n}_zone_x'] = -1
                        history[f'prev_{n}_zone_y'] = -1
                        history[f'prev_{n}_success'] = -1
                
                # Look up spectral embeddings
                start_zone = f"Z_{curr_x}_{curr_y}"
                z_emb = zone_embeddings.get(start_zone, np.zeros(4))
                p_emb = player_embeddings.get(player, np.zeros(8))
                m_emb = manager_embeddings.get(mgr_name, np.zeros(4))
                
                # History zone embeddings
                h_embs = {}
                for n in range(1, 3):
                    px = history[f'prev_{n}_zone_x']
                    py = history[f'prev_{n}_zone_y']
                    if px != -1 and py != -1:
                        prev_z_name = f"Z_{px}_{py}"
                        pz_emb = zone_embeddings.get(prev_z_name, np.zeros(4))
                    else:
                        pz_emb = np.zeros(4)
                    for d in range(4):
                        h_embs[f'prev_{n}_zone_emb_{d}'] = pz_emb[d]
                        
                # Get player role
                pos = row_dict.get('position', 'Center Midfield')
                player_role = ROLE_MAPPING.get(pos, 2)
                
                record = {
                    'start_zone_x': curr_x,
                    'start_zone_y': curr_y,
                    'passer_accuracy': passer_acc,
                    'passer_progressive_ratio': passer_prog,
                    'opp_defensive_rate': opp_def_rate,
                    'opp_gk_save_ratio': opp_gk_save,
                    'manager_directness': mgr_dir,
                    'manager_width': mgr_wid,
                    'score_differential': p['score_differential'],
                    'possession_duration': p['possession_duration'],
                    'pass_sequence_index': i,
                    'pass_length': pass_length,
                    'pass_angle': pass_angle,
                    'player_role': player_role,
                    'prev_pass_direction_1': prev_dir_1,
                    'prev_pass_direction_2': prev_dir_2,
                    'prev_pass_direction_3': prev_dir_3,
                    **history,
                    'outcome': outcome,
                    'end_zone_x': end_x,
                    'end_zone_y': end_y
                }
                
                # Add opponent defensive density for all 30 target zones
                for tx in range(6):
                    for ty in range(5):
                        t_zone = f"Z_{tx}_{ty}"
                        record[f'target_def_density_{tx}_{ty}'] = def_profiles.get((opp_team, t_zone), 0.0)
                
                # Add embeddings
                for d in range(4):
                    record[f'zone_emb_{d}'] = z_emb[d]
                for d in range(8):
                    record[f'player_emb_{d}'] = p_emb[d]
                for d in range(4):
                    record[f'manager_emb_{d}'] = m_emb[d]
                record.update(h_embs)
                
                dataset.append(record)
                
        if (idx + 1) % 100 == 0 or (idx + 1) == len(match_files):
            print(f"  Processed {idx + 1} matches...")
            
    # Resolve player roles and save
    import json
    player_roles = {}
    for p, positions in player_positions.items():
        most_common_pos = max(positions, key=positions.get)
        player_roles[p] = ROLE_MAPPING.get(most_common_pos, 2)
    os.makedirs("data/models", exist_ok=True)
    with open("data/models/player_roles.json", "w") as f:
        json.dump(player_roles, f)
    print("Saved player roles to data/models/player_roles.json")
            
    df_dataset = pd.DataFrame(dataset)
    os.makedirs("data", exist_ok=True)
    csv_filename = f"data/ml_training_data_{mode}.csv"
    df_dataset.to_csv(csv_filename, index=False)
    print(f"Saved training dataset with {len(df_dataset)} rows to {csv_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='iteration', choices=['iteration', 'production'])
    args = parser.parse_args()
    
    extract_features_and_targets(args.mode)
