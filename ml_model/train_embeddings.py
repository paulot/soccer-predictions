import os
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from utils import parse_location, map_coordinates_to_zone, TEAM_TO_MANAGER
import click
from typing import Dict, List, Any

def train_all_embeddings(mode: str = 'iteration') -> None:
    print(f"Training Spectral Embeddings (Mode: {mode.upper()})...")
    
    raw_dir: str = "data/raw_events"
    match_files: List[str] = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    if mode == 'iteration':
        match_files = match_files[:50]
        
    zones: List[str] = [f"Z_{x}_{y}" for x in range(6) for y in range(5)]
    zone_to_idx: Dict[str, int] = {z: i for i, z in enumerate(zones)}
    
    # Initialize matrices
    # 1. Zone-Zone transition matrix
    zone_transition: np.ndarray = np.zeros((30, 30))
    
    # 2. Player activity matrix: player -> [start_zone_counts (30), end_zone_counts (30), success_rate (1)]
    player_stats: Dict[str, Dict[str, Any]] = {}
    
    # 3. Manager activity matrix: manager -> [start_zone_counts (30), end_zone_counts (30)]
    manager_stats: Dict[str, Dict[str, Any]] = {}
    
    for filename in match_files:
        match_path: str = os.path.join(raw_dir, filename)
        try:
            df_events: pd.DataFrame = pd.read_csv(match_path)
        except Exception:
            continue
            
        passes: pd.DataFrame = df_events[df_events['type'] == 'Pass'].copy()
        if passes.empty:
            continue
            
        passes['loc_parsed'] = passes['location'].apply(parse_location)
        passes['pass_end_loc_parsed'] = passes['pass_end_location'].apply(parse_location)
        
        for _, row in passes.iterrows():
            loc = row['loc_parsed']
            end_loc = row['pass_end_loc_parsed']
            if not loc or not end_loc or len(loc) < 2 or len(end_loc) < 2:
                continue
                
            sz: str = map_coordinates_to_zone(loc[0], loc[1])
            ez: str = map_coordinates_to_zone(end_loc[0], end_loc[1])
            outcome: int = 1 if pd.notnull(row.get('pass_outcome')) else 0 # 1 = Turnover, 0 = Success
            
            s_idx: int = zone_to_idx[sz]
            e_idx: int = zone_to_idx[ez]
            
            # Update Zone Transition (only successful passes)
            if outcome == 0:
                zone_transition[s_idx, e_idx] += 1
                
            # Update Player Stats
            player: str = row['player']
            if pd.notnull(player):
                if player not in player_stats:
                    player_stats[player] = {
                        'start': np.zeros(30),
                        'end': np.zeros(30),
                        'successes': 0,
                        'total': 0
                    }
                player_stats[player]['start'][s_idx] += 1
                if outcome == 0:
                    player_stats[player]['end'][e_idx] += 1
                    player_stats[player]['successes'] += 1
                player_stats[player]['total'] += 1
                
            # Update Manager Stats
            team: str = row['team']
            mgr_name: str = TEAM_TO_MANAGER.get(team)
            if mgr_name:
                if mgr_name not in manager_stats:
                    manager_stats[mgr_name] = {
                        'start': np.zeros(30),
                        'end': np.zeros(30)
                    }
                manager_stats[mgr_name]['start'][s_idx] += 1
                if outcome == 0:
                    manager_stats[mgr_name]['end'][e_idx] += 1
                    
    # --- 1. Compute Zone Embeddings (D=4) ---
    print("Computing Zone Embeddings...")
    # Row normalize transition matrix to get probabilities
    row_sums: np.ndarray = zone_transition.sum(axis=1, keepdims=True)
    zone_prob_matrix: np.ndarray = np.divide(zone_transition, row_sums, out=np.zeros_like(zone_transition), where=row_sums!=0)
    
    svd_zone: TruncatedSVD = TruncatedSVD(n_components=4, random_state=42)
    zone_emb_matrix: np.ndarray = svd_zone.fit_transform(zone_prob_matrix)
    zone_embeddings: Dict[str, np.ndarray] = {zones[i]: zone_emb_matrix[i] for i in range(30)}
    
    # --- 2. Compute Player Embeddings (K=8) ---
    print("Computing Player Embeddings...")
    player_names: List[str] = list(player_stats.keys())
    player_matrix: List[np.ndarray] = []
    for p in player_names:
        stats = player_stats[p]
        tot = stats['total'] if stats['total'] > 0 else 1
        # Normalize start/end distributions
        start_dist = stats['start'] / tot
        end_dist = stats['end'] / max(1, stats['successes'])
        acc = stats['successes'] / tot
        
        feature_vector = np.concatenate([start_dist, end_dist, [acc]])
        player_matrix.append(feature_vector)
        
    player_matrix_arr: np.ndarray = np.array(player_matrix)
    # We want at most min(features, players) components
    n_components_p: int = min(8, len(player_names))
    svd_player: TruncatedSVD = TruncatedSVD(n_components=n_components_p, random_state=42)
    player_emb_matrix: np.ndarray = svd_player.fit_transform(player_matrix_arr)
    
    # Pad if fewer than 8 components
    if n_components_p < 8:
        padding = np.zeros((len(player_names), 8 - n_components_p))
        player_emb_matrix = np.hstack([player_emb_matrix, padding])
        
    player_embeddings: Dict[str, np.ndarray] = {player_names[i]: player_emb_matrix[i] for i in range(len(player_names))}
    
    # --- 3. Compute Manager Embeddings (T=4) ---
    print("Computing Manager Embeddings...")
    manager_names: List[str] = list(manager_stats.keys())
    manager_matrix: List[np.ndarray] = []
    for m in manager_names:
        stats = manager_stats[m]
        tot_start = stats['start'].sum() if stats['start'].sum() > 0 else 1
        tot_end = stats['end'].sum() if stats['end'].sum() > 0 else 1
        
        start_dist = stats['start'] / tot_start
        end_dist = stats['end'] / tot_end
        
        feature_vector = np.concatenate([start_dist, end_dist])
        manager_matrix.append(feature_vector)
        
    manager_matrix_arr: np.ndarray = np.array(manager_matrix)
    n_components_m: int = min(4, len(manager_names))
    svd_manager: TruncatedSVD = TruncatedSVD(n_components=n_components_m, random_state=42)
    manager_emb_matrix: np.ndarray = svd_manager.fit_transform(manager_matrix_arr)
    
    if n_components_m < 4:
        padding = np.zeros((len(manager_names), 4 - n_components_m))
        manager_emb_matrix = np.hstack([manager_emb_matrix, padding])
        
    manager_embeddings: Dict[str, np.ndarray] = {manager_names[i]: manager_emb_matrix[i] for i in range(len(manager_names))}
    
    # Save embeddings
    os.makedirs("data/embeddings", exist_ok=True)
    with open("data/embeddings/zone_embeddings.pkl", 'wb') as f:
        pickle.dump(zone_embeddings, f)
    with open("data/embeddings/player_embeddings.pkl", 'wb') as f:
        pickle.dump(player_embeddings, f)
    with open("data/embeddings/manager_embeddings.pkl", 'wb') as f:
        pickle.dump(manager_embeddings, f)
        
    print("Saved all embeddings to data/embeddings/")

@click.command(help="Train spectral embeddings for zones, players, and managers.")
@click.option('--mode', type=click.Choice(['iteration', 'production']), default='iteration', help='Mode: iteration or production')
def main(mode: str) -> None:
    train_all_embeddings(mode)

if __name__ == "__main__":
    main()
