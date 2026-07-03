import os
import click
import pandas as pd
import numpy as np
import pickle
import json
from utils import parse_location, map_coordinates_to_zone, TEAM_TO_MANAGER
from typing import Dict, List, Any, Optional, Tuple, Set

def parse_timestamp_to_seconds(ts_str: Any) -> float:
    if pd.isna(ts_str): 
        return 0.0
    try:
        parts: List[str] = str(ts_str).split(':')
        h: int = int(parts[0])
        m: int = int(parts[1])
        s: float = float(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return 0.0

def extract_features_and_targets(mode: str = 'iteration') -> None:
    print(f"Extracting features for ML Transition Model (Mode: {mode.upper()})...")
    
    # 1. Load profiles for joining
    try:
        df_players: pd.DataFrame = pd.read_csv("data/statsbomb_player_profiles.csv")
        player_profiles: Dict[str, Dict[str, float]] = df_players.set_index('player').to_dict(orient='index')
        
        df_gk: pd.DataFrame = pd.read_csv("data/goalkeeper_profiles.csv")
        gk_profiles: Dict[str, Dict[str, float]] = df_gk.set_index('goalkeeper').to_dict(orient='index')
        
        df_def: pd.DataFrame = pd.read_csv("data/team_defensive_profiles.csv")
        # Map (team, zone) -> defensive_rate
        def_profiles: Dict[Tuple[str, str], float] = {}
        for _, row in df_def.iterrows():
            def_profiles[(str(row['team']), str(row['zone']))] = float(row['defensive_rate'])
            
        df_mgr: pd.DataFrame = pd.read_csv("data/manager_profiles.csv")
        mgr_profiles: Dict[str, Dict[str, float]] = df_mgr.set_index('manager').to_dict(orient='index')
    except Exception as e:
        print(f"Error loading profiles: {e}. Please run download_data.py first.")
        return

    # Load spectral embeddings
    emb_dir: str = "data/embeddings"
    if not os.path.exists(os.path.join(emb_dir, "zone_embeddings.pkl")):
        print("  Embeddings not found. Training spectral embeddings first...")
        from ml_model.train_embeddings import train_all_embeddings
        train_all_embeddings(mode)
        
    with open(os.path.join(emb_dir, "zone_embeddings.pkl"), 'rb') as f:
        zone_embeddings: Dict[str, np.ndarray] = pickle.load(f)
    with open(os.path.join(emb_dir, "player_embeddings.pkl"), 'rb') as f:
        player_embeddings: Dict[str, np.ndarray] = pickle.load(f)
    with open(os.path.join(emb_dir, "manager_embeddings.pkl"), 'rb') as f:
        manager_embeddings: Dict[str, np.ndarray] = pickle.load(f)

    # TEAM_TO_MANAGER is imported from utils

    raw_dir: str = "data/raw_events"
    if not os.path.exists(raw_dir):
        print(f"Raw events directory {raw_dir} not found. Please run download_data.py first.")
        return
        
    match_files: List[str] = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    
    # In iteration mode, only use a small subset of matches (e.g. 50)
    if mode == 'iteration':
        match_files = match_files[:50]
        print(f"  Iteration Mode: Limiting feature extraction to first {len(match_files)} matches.")
    
    dataset: List[Dict[str, Any]] = []
    player_positions: Dict[str, Dict[str, int]] = {}
    
    ROLE_MAPPING: Dict[str, int] = {
        'Goalkeeper': 0,
        'Right Back': 1, 'Left Back': 1, 'Center Back': 1, 'Right Center Back': 1, 'Left Center Back': 1,
        'Right Wing Back': 1, 'Left Wing Back': 1,
        'Center Defensive Midfield': 2, 'Right Center Midfield': 2, 'Left Center Midfield': 2,
        'Right Midfield': 2, 'Left Midfield': 2, 'Center Midfield': 2, 'Center Attacking Midfield': 2,
        'Right Wing': 3, 'Left Wing': 3, 'Center Forward': 3, 'Secondary Striker': 3,
        'Right Center Forward': 3, 'Left Center Forward': 3
    }
    
    for idx, filename in enumerate(match_files):
        match_path: str = os.path.join(raw_dir, filename)
        try:
            df_events: pd.DataFrame = pd.read_csv(match_path)
        except Exception:
            continue
            
        # Sort chronologically
        df_events = df_events.sort_values(by=['period', 'timestamp'])
        
        home_team: str = str(df_events['team'].dropna().unique()[0])
        away_team: str = str(df_events['team'].dropna().unique()[1])
        
        # Track running score
        home_score: int = 0
        away_score: int = 0
        
        # Track possession start times
        poss_start_times: Dict[int, float] = {}
        for poss_id, gp in df_events.groupby('possession'):
            first_event = gp.iloc[0]
            ts = first_event.get('timestamp')
            poss_start_times[int(poss_id)] = parse_timestamp_to_seconds(ts)
            
        # Accumulate player positions
        df_pos = df_events.dropna(subset=['player', 'position'])
        for _, row in df_pos.iterrows():
            p: str = str(row['player'])
            pos: str = str(row['position'])
            if p not in player_positions:
                player_positions[p] = {}
            player_positions[p][pos] = player_positions[p].get(pos, 0) + 1
            
        # Find all goalkeepers
        player_to_team: Dict[str, str] = df_events.dropna(subset=['player', 'team']).set_index('player')['team'].to_dict()
        home_gk: Optional[str] = None
        away_gk: Optional[str] = None
        for player, team in player_to_team.items():
            if player in gk_profiles:
                if team == home_team:
                    home_gk = player
                elif team == away_team:
                    away_gk = player
                    
        # Extract passes with score and time context chronologically
        passes_with_context: List[Dict[str, Any]] = []
        for _, row in df_events.iterrows():
            etype: str = str(row['type'])
            team: str = str(row['team'])
            
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
                    
                pass_time: float = parse_timestamp_to_seconds(row.get('timestamp'))
                poss_id: int = int(row['possession'])
                poss_start: float = poss_start_times.get(poss_id, pass_time)
                duration: float = max(0.0, pass_time - poss_start)
                
                score_diff: int = home_score - away_score if team == home_team else away_score - home_score
                
                period: int = int(row.get('period', 1))
                time_ratio: float = (period - 1) * 0.5 + min(1.0, pass_time / 2700.0) * 0.5
                under_pressure: int = 1 if row.get('under_pressure') == True or row.get('under_pressure') == 1.0 else 0
                
                passes_with_context.append({
                    'event_row': row.to_dict(),
                    'loc_parsed': loc,
                    'pass_end_loc_parsed': end_loc,
                    'score_differential': score_diff,
                    'possession_duration': duration,
                    'time_ratio': time_ratio,
                    'under_pressure': under_pressure
                })
                
        if not passes_with_context:
            continue
            
        # Group passes by possession to build the sequence history
        df_passes_context: pd.DataFrame = pd.DataFrame(passes_with_context)
        for _, group in df_passes_context.groupby(lambda idx: passes_with_context[idx]['event_row']['possession']):
            poss_passes: List[Dict[str, Any]] = group.to_dict(orient='records')
            
            possession_directions: List[int] = []
            
            for i, p in enumerate(poss_passes):
                row_dict: Dict[str, Any] = p['event_row']
                loc = p['loc_parsed']
                end_loc = p['pass_end_loc_parsed']
                
                curr_x: int = min(int(loc[0] / 20), 5)
                curr_y: int = min(int(loc[1] / 16), 4)
                end_x: int = min(int(end_loc[0] / 20), 5)
                end_y: int = min(int(end_loc[1] / 16), 4)
                
                player: str = str(row_dict['player'])
                team: str = str(row_dict['team'])
                opp_team: str = away_team if team == home_team else home_team
                opp_gk: Optional[str] = away_gk if team == home_team else home_gk
                
                outcome: int = 1 if pd.notnull(row_dict.get('pass_outcome')) else 0
                
                p_profile = player_profiles.get(player, {'accuracy': 0.80, 'progressive_ratio': 0.25})
                passer_acc: float = p_profile['accuracy']
                passer_prog: float = p_profile['progressive_ratio']
                
                start_zone: str = f"Z_{curr_x}_{curr_y}"
                opp_def_rate: float = def_profiles.get((opp_team, start_zone), 0.0)
                opp_gk_profile = gk_profiles.get(opp_gk or "", {'save_ratio': 0.70})
                opp_gk_save: float = opp_gk_profile['save_ratio']
                
                mgr_name: str = TEAM_TO_MANAGER.get(team, "")
                m_profile = mgr_profiles.get(mgr_name, {'directness': 5.0, 'width': 5.0})
                mgr_dir: float = m_profile['directness']
                mgr_wid: float = m_profile['width']
                
                # Calculate spatial features based on discrete zone centers
                start_cx: float = curr_x * 20.0 + 10.0
                start_cy: float = curr_y * 16.0 + 8.0
                end_cx: float = end_x * 20.0 + 10.0
                end_cy: float = end_y * 16.0 + 8.0
                dx: float = end_cx - start_cx
                dy: float = end_cy - start_cy
                pass_length: float = float(np.sqrt(dx**2 + dy**2))
                pass_angle: float = float(np.arctan2(dy, dx))
                
                # Direction of this pass
                pass_dx: int = end_x - curr_x
                direction: int = 1 if pass_dx > 0 else (-1 if pass_dx < 0 else 0)
                
                # Extract history of last 3 passes in this possession
                prev_dir_1: int = possession_directions[-1] if len(possession_directions) >= 1 else 0
                prev_dir_2: int = possession_directions[-2] if len(possession_directions) >= 2 else 0
                prev_dir_3: int = possession_directions[-3] if len(possession_directions) >= 3 else 0
                
                possession_directions.append(direction)
                
                # History features (N=2)
                history: Dict[str, float] = {}
                for n in range(1, 3):
                    if i - n >= 0:
                        prev_p = poss_passes[i - n]
                        prev_loc = prev_p['loc_parsed']
                        if prev_loc and len(prev_loc) >= 2:
                            px: int = min(int(prev_loc[0] / 20), 5)
                            py: int = min(int(prev_loc[1] / 16), 4)
                        else:
                            px, py = -1, -1
                        p_outcome: int = 1 if pd.notnull(prev_p['event_row'].get('pass_outcome')) else 0
                        
                        history[f'prev_{n}_zone_x'] = float(px)
                        history[f'prev_{n}_zone_y'] = float(py)
                        history[f'prev_{n}_success'] = float(1 - p_outcome)
                    else:
                        history[f'prev_{n}_zone_x'] = -1.0
                        history[f'prev_{n}_zone_y'] = -1.0
                        history[f'prev_{n}_success'] = -1.0
                
                # Look up spectral embeddings
                z_emb: np.ndarray = zone_embeddings.get(start_zone, np.zeros(4))
                p_emb: np.ndarray = player_embeddings.get(player, np.zeros(8))
                m_emb: np.ndarray = manager_embeddings.get(mgr_name, np.zeros(4))
                
                # History zone embeddings
                h_embs: Dict[str, float] = {}
                for n in range(1, 3):
                    px_f = history[f'prev_{n}_zone_x']
                    py_f = history[f'prev_{n}_zone_y']
                    if px_f != -1.0 and py_f != -1.0:
                        prev_z_name = f"Z_{int(px_f)}_{int(py_f)}"
                        pz_emb: np.ndarray = zone_embeddings.get(prev_z_name, np.zeros(4))
                    else:
                        pz_emb = np.zeros(4)
                    for d in range(4):
                        h_embs[f'prev_{n}_zone_emb_{d}'] = float(pz_emb[d])
                        
                # Get player role
                pos: str = str(row_dict.get('position', 'Center Midfield'))
                player_role: int = ROLE_MAPPING.get(pos, 2)
                
                game_state_momentum: float = p['score_differential'] * (1.0 + p['time_ratio'])
                end_def_rate: float = def_profiles.get((opp_team, f"Z_{end_x}_{end_y}"), 0.0)
                pressure_differential: float = end_def_rate - opp_def_rate
                
                record: Dict[str, Any] = {
                    'start_zone_x': float(curr_x),
                    'start_zone_y': float(curr_y),
                    'passer_accuracy': passer_acc,
                    'passer_progressive_ratio': passer_prog,
                    'opp_defensive_rate': opp_def_rate,
                    'opp_gk_save_ratio': opp_gk_save,
                    'manager_directness': mgr_dir,
                    'manager_width': mgr_wid,
                    'score_differential': float(p['score_differential']),
                    'possession_duration': p['possession_duration'],
                    'pass_sequence_index': float(i),
                    'pass_length': pass_length,
                    'pass_angle': pass_angle,
                    'player_role': float(player_role),
                    'prev_pass_direction_1': float(prev_dir_1),
                    'prev_pass_direction_2': float(prev_dir_2),
                    'prev_pass_direction_3': float(prev_dir_3),
                    'under_pressure': float(p['under_pressure']),
                    'game_state_momentum': game_state_momentum,
                    'pressure_differential': pressure_differential,
                    **history,
                    'outcome': float(outcome),
                    'end_zone_x': float(end_x),
                    'end_zone_y': float(end_y)
                }
                
                # Add opponent defensive density for all 30 target zones
                for tx in range(6):
                    for ty in range(5):
                        t_zone = f"Z_{tx}_{ty}"
                        record[f'target_def_density_{tx}_{ty}'] = def_profiles.get((opp_team, t_zone), 0.0)
                
                # Add embeddings
                for d in range(4):
                    record[f'zone_emb_{d}'] = float(z_emb[d])
                for d in range(8):
                    record[f'player_emb_{d}'] = float(p_emb[d])
                for d in range(4):
                    record[f'manager_emb_{d}'] = float(m_emb[d])
                record.update(h_embs)
                
                dataset.append(record)
                
        if (idx + 1) % 100 == 0 or (idx + 1) == len(match_files):
            print(f"  Processed {idx + 1} matches...")
            
    # Resolve player roles and save
    player_roles: Dict[str, int] = {}
    for p, positions in player_positions.items():
        most_common_pos = max(positions, key=positions.get)
        player_roles[p] = ROLE_MAPPING.get(most_common_pos, 2)
    os.makedirs("data/models", exist_ok=True)
    with open("data/models/player_roles.json", "w") as f:
        json.dump(player_roles, f)
    print("Saved player roles to data/models/player_roles.json")
            
    df_dataset: pd.DataFrame = pd.DataFrame(dataset)
    os.makedirs("data", exist_ok=True)
    csv_filename: str = f"data/ml_training_data_{mode}.csv"
    df_dataset.to_csv(csv_filename, index=False)
    print(f"Saved training dataset with {len(df_dataset)} rows to {csv_filename}")

@click.command(help="Extract features and targets from StatsBomb raw events.")
@click.option('--mode', type=click.Choice(['iteration', 'production']), default='iteration', help='Mode: iteration or production')
def main(mode: str) -> None:
    extract_features_and_targets(mode)

if __name__ == "__main__":
    main()
