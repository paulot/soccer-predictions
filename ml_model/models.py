import os
import pickle
import json
import numpy as np
import pandas as pd
import torch
from ml_model.pytorch_models import OutcomeNN, DestinationNN

class BaseTransitionModel:
    """
    Abstract Base Class for MCMC Transition Models.
    All models must implement get_transition_probabilities and get_turnover_probability.
    """
    def get_transition_probabilities(self, current_zone, player_on_ball, player_profiles, 
                                      manager_profiles, team_to_manager, player_to_team, 
                                      home_team, away_team, zones, **kwargs):
        raise NotImplementedError
        
    def get_turnover_probability(self, current_zone, player_on_ball, player_profiles, 
                                  team_defensive_profiles, player_to_team, home_team, away_team, **kwargs):
        raise NotImplementedError
 
 
class HeuristicTransitionModel(BaseTransitionModel):
    """
    Wraps our Phase 5 Recency-Weighted Heuristic Model.
    """
    def __init__(self, base_matrix):
        self.base_matrix = base_matrix
 
    def get_transition_probabilities(self, current_zone, player_on_ball, player_profiles, 
                                      manager_profiles, team_to_manager, player_to_team, 
                                      home_team, away_team, zones, **kwargs):
        # Get baseline transition probabilities
        zone_probs = self.base_matrix.loc[current_zone].copy()
        if zone_probs.sum() == 0:
            return zone_probs
            
        # 1. Apply Player Modifiers
        from mcmc_simulation import apply_player_modifier
        if player_on_ball and player_on_ball in player_profiles:
            zone_probs = apply_player_modifier(zone_probs, player_profiles[player_on_ball], current_zone, zones)
            
        # 2. Apply Manager Tactical Modifiers (Directness and Width)
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        mgr_name = team_to_manager.get(current_team)
        mgr = manager_profiles.get(mgr_name, {"directness": 5, "width": 5, "tempo": 5})
        
        start_x = int(current_zone.split('_')[1])
        for zone in zones:
            end_x = int(zone.split('_')[1])
            end_y = int(zone.split('_')[2])
            
            # Directness
            dist_x = end_x - start_x
            if dist_x > 1:
                zone_probs[zone] *= (1.0 + (mgr['directness'] - 5) * 0.12)
            elif dist_x == 0 or dist_x == -1:
                zone_probs[zone] *= (1.0 - (mgr['directness'] - 5) * 0.04)
                
            # Width
            if end_y in [0, 4]:
                zone_probs[zone] *= (1.0 + (mgr['width'] - 5) * 0.08)
            else:
                zone_probs[zone] *= (1.0 - (mgr['width'] - 5) * 0.04)
                
        # Re-normalize
        if zone_probs.sum() > 0:
            zone_probs = zone_probs / zone_probs.sum()
        return zone_probs

    def get_turnover_probability(self, current_zone, player_on_ball, player_profiles, 
                                  team_defensive_profiles, player_to_team, home_team, away_team, **kwargs):
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        defending_team = away_team if current_team == home_team else home_team
        def_rate = team_defensive_profiles.get(defending_team, {}).get(current_zone, 0.0)
        def_factor = min(0.15, def_rate * 0.03)
        
        if player_on_ball and player_on_ball in player_profiles:
            return max(0.05, min(0.30, (1.0 - player_profiles[player_on_ball]['accuracy']) * 0.5 + 0.05 + def_factor))
        else:
            return 0.12 + def_factor


class MLTransitionModel(BaseTransitionModel):
    """
    Uses trained Machine Learning models (e.g. XGBoost, Random Forest)
    to dynamically generate transition and turnover probabilities with lazy caching.
    """
    def __init__(self, outcome_model_path, destination_model_path):
        with open(outcome_model_path, 'rb') as f:
            self.outcome_model = pickle.load(f)
        with open(destination_model_path, 'rb') as f:
            self.destination_model = pickle.load(f)
            
        # Load spectral embeddings
        import pickle as pkl
        with open("data/embeddings/zone_embeddings.pkl", 'rb') as f:
            self.zone_embeddings = pkl.load(f)
        with open("data/embeddings/player_embeddings.pkl", 'rb') as f:
            self.player_embeddings = pkl.load(f)
        with open("data/embeddings/manager_embeddings.pkl", 'rb') as f:
            self.manager_embeddings = pkl.load(f)
            
        # Load goalkeeper profiles
        try:
            df_gk = pd.read_csv("data/goalkeeper_profiles.csv")
            self.gk_profiles = df_gk.set_index('goalkeeper').to_dict(orient='index')
        except:
            self.gk_profiles = {}
            
        # Load team defensive profiles
        try:
            df_def = pd.read_csv("data/team_defensive_profiles.csv")
            self.def_profiles = {}
            for _, row in df_def.iterrows():
                self.def_profiles[(row['team'], row['zone'])] = row['defensive_rate']
        except:
            self.def_profiles = {}
            
        # Load player roles
        try:
            with open("data/models/player_roles.json", 'r') as f:
                self.player_roles = json.load(f)
        except:
            self.player_roles = {}
            
        # Load PyTorch scalers
        try:
            with open("data/models/neural_network_outcome_scaler.pkl", 'rb') as f:
                self.outcome_scaler = pkl.load(f)
            with open("data/models/neural_network_destination_scaler.pkl", 'rb') as f:
                self.destination_scaler = pkl.load(f)
        except:
            self.outcome_scaler = None
            self.destination_scaler = None
            
        # Caches to avoid expensive predict_proba calls in the MCMC loop
        self.cache_probs = {}
        self.cache_turnover = {}
            
    def _compile_features(self, current_zone, player_on_ball, player_profiles, 
                          manager_profiles, team_to_manager, player_to_team, 
                          home_team, away_team, history=None, score_differential=0, possession_duration=0.0,
                          pass_sequence_index=0, pass_length=0.0, pass_angle=0.0, possession_directions=None,
                          time_ratio=0.0, under_pressure=0, next_zone=None):
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        defending_team = away_team if current_team == home_team else home_team
        mgr_name = team_to_manager.get(current_team)
        
        # Look up embeddings
        z_emb = self.zone_embeddings.get(current_zone, np.zeros(4))
        p_emb = self.player_embeddings.get(player_on_ball, np.zeros(8))
        m_emb = self.manager_embeddings.get(mgr_name, np.zeros(4))
        
        # Look up defensive pressure in current_zone
        opp_def_rate = self.def_profiles.get((defending_team, current_zone), 0.0)
        
        # Look up opponent goalkeeper and their save ratio
        opp_gk = None
        for p, t in player_to_team.items():
            if t == defending_team and p in self.gk_profiles:
                opp_gk = p
                break
        opp_gk_save = self.gk_profiles.get(opp_gk, {'save_ratio': 0.70}).get('save_ratio', 0.70)
        
        if history is None:
            history = [(-1, -1, -1), (-1, -1, -1)]
        p1, p2 = history[0], history[1]
        
        # Look up history zone embeddings
        if p1[0] != -1:
            p1_z_name = f"Z_{p1[0]}_{p1[1]}"
            p1_emb = self.zone_embeddings.get(p1_z_name, np.zeros(4))
        else:
            p1_emb = np.zeros(4)
            
        if p2[0] != -1:
            p2_z_name = f"Z_{p2[0]}_{p2[1]}"
            p2_emb = self.zone_embeddings.get(p2_z_name, np.zeros(4))
        else:
            p2_emb = np.zeros(4)
            
        # Get player role
        player_role = self.player_roles.get(player_on_ball, 2)
        
        # Get previous pass directions
        if possession_directions is None:
            possession_directions = []
        prev_dir_1 = possession_directions[-1] if len(possession_directions) >= 1 else 0
        prev_dir_2 = possession_directions[-2] if len(possession_directions) >= 2 else 0
        prev_dir_3 = possession_directions[-3] if len(possession_directions) >= 3 else 0
        
        # Calculate game state momentum
        game_state_momentum = score_differential * (1.0 + time_ratio)
        
        # Calculate pressure differential if next_zone is provided
        if next_zone:
            end_def_rate = self.def_profiles.get((defending_team, next_zone), 0.0)
            pressure_differential = end_def_rate - opp_def_rate
        else:
            pressure_differential = 0.0
        
        feats = {
            'start_zone_x': int(current_zone.split('_')[1]),
            'start_zone_y': int(current_zone.split('_')[2]),
            'opp_defensive_rate': opp_def_rate,
            'opp_gk_save_ratio': opp_gk_save,
            'score_differential': score_differential,
            'possession_duration': possession_duration,
            'pass_sequence_index': pass_sequence_index,
            'pass_length': pass_length,
            'pass_angle': pass_angle,
            'player_role': player_role,
            'prev_pass_direction_1': prev_dir_1,
            'prev_pass_direction_2': prev_dir_2,
            'prev_pass_direction_3': prev_dir_3,
            'under_pressure': under_pressure,
            'game_state_momentum': game_state_momentum,
            'pressure_differential': pressure_differential,
            'prev_1_success': p1[2],
            'prev_2_success': p2[2]
        }
        
        # Add opponent defensive density for all 30 target zones
        for tx in range(6):
            for ty in range(5):
                t_zone = f"Z_{tx}_{ty}"
                feats[f'target_def_density_{tx}_{ty}'] = self.def_profiles.get((defending_team, t_zone), 0.0)
        
        # Add embeddings
        for d in range(4):
            feats[f'zone_emb_{d}'] = z_emb[d]
        for d in range(8):
            feats[f'player_emb_{d}'] = p_emb[d]
        for d in range(4):
            feats[f'manager_emb_{d}'] = m_emb[d]
        for d in range(4):
            feats[f'prev_1_zone_emb_{d}'] = p1_emb[d]
            feats[f'prev_2_zone_emb_{d}'] = p2_emb[d]
            
        return feats

    def get_transition_probabilities(self, current_zone, player_on_ball, player_profiles, 
                                      manager_profiles, team_to_manager, player_to_team, 
                                      home_team, away_team, zones, history=None, score_differential=0, possession_duration=0.0,
                                      pass_sequence_index=0, possession_directions=None, time_ratio=0.0, under_pressure=0):
        if history is None:
            history = [(-1, -1, -1), (-1, -1, -1)]
        hist_tuple = tuple(history)
        pd_tuple = tuple(possession_directions) if possession_directions else ()
        cache_key = (player_on_ball, current_zone, hist_tuple, score_differential, round(possession_duration, 1), pass_sequence_index, pd_tuple, time_ratio, under_pressure)
        if cache_key in self.cache_probs:
            return self.cache_probs[cache_key]
            
        feats = self._compile_features(current_zone, player_on_ball, player_profiles, 
                                       manager_profiles, team_to_manager, player_to_team, 
                                       home_team, away_team, history, score_differential, possession_duration,
                                       pass_sequence_index, 0.0, 0.0, possession_directions,
                                       time_ratio, under_pressure, None)
        
        df_feats = pd.DataFrame([feats])
        dest_features = [
            'start_zone_x', 'start_zone_y',
            'zone_emb_0', 'zone_emb_1', 'zone_emb_2', 'zone_emb_3',
            'player_emb_0', 'player_emb_1', 'player_emb_2', 'player_emb_3',
            'player_emb_4', 'player_emb_5', 'player_emb_6', 'player_emb_7',
            'opp_defensive_rate', 'opp_gk_save_ratio',
            'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
            'score_differential', 'possession_duration', 'pass_sequence_index',
            'player_role',
            'prev_pass_direction_1', 'prev_pass_direction_2', 'prev_pass_direction_3',
            'under_pressure', 'game_state_momentum',
            'prev_1_zone_emb_0', 'prev_1_zone_emb_1', 'prev_1_zone_emb_2', 'prev_1_zone_emb_3', 'prev_1_success',
            'prev_2_zone_emb_0', 'prev_2_zone_emb_1', 'prev_2_zone_emb_2', 'prev_2_zone_emb_3', 'prev_2_success'
        ] + [f'target_def_density_{tx}_{ty}' for tx in range(6) for ty in range(5)]
        df_feats = df_feats[dest_features]
        
        if isinstance(self.destination_model, torch.nn.Module):
            with torch.no_grad():
                continuous_cols = [c for c in df_feats.columns if c not in ['player_role', 'start_zone_x', 'start_zone_y']]
                df_feats_scaled = df_feats.copy()
                df_feats_scaled[continuous_cols] = self.destination_scaler.transform(df_feats[continuous_cols])
                feats_tensor = torch.FloatTensor(df_feats_scaled.values)
                logits = self.destination_model(feats_tensor)
                dest_probs = torch.softmax(logits, dim=1).numpy()[0]
        else:
            dest_probs = self.destination_model.predict_proba(df_feats)[0]
        
        zone_probs = pd.Series(0.0, index=zones)
        if isinstance(self.destination_model, torch.nn.Module):
            for idx, prob in enumerate(dest_probs):
                z_x = idx // 5
                z_y = idx % 5
                zone_name = f"Z_{z_x}_{z_y}"
                zone_probs[zone_name] = prob
        else:
            for idx, prob in enumerate(dest_probs):
                class_val = self.destination_model.classes_[idx]
                if isinstance(class_val, (int, np.integer)):
                    z_x = class_val // 5
                    z_y = class_val % 5
                    zone_name = f"Z_{z_x}_{z_y}"
                else:
                    zone_name = class_val
                zone_probs[zone_name] = prob
            
        if zone_probs.sum() > 0:
            zone_probs = zone_probs / zone_probs.sum()
            
        self.cache_probs[cache_key] = zone_probs
        return zone_probs

    def get_turnover_probability(self, current_zone, player_on_ball, player_profiles, 
                                  team_defensive_profiles, player_to_team, home_team, away_team, 
                                  history=None, score_differential=0, possession_duration=0.0,
                                  pass_sequence_index=0, next_zone=None, possession_directions=None,
                                  time_ratio=0.0, under_pressure=0):
        if next_zone is None:
            next_zone = current_zone
            
        if history is None:
            history = [(-1, -1, -1), (-1, -1, -1)]
        hist_tuple = tuple(history)
        pd_tuple = tuple(possession_directions) if possession_directions else ()
        cache_key = (player_on_ball, current_zone, next_zone, hist_tuple, score_differential, round(possession_duration, 1), pass_sequence_index, pd_tuple, time_ratio, under_pressure)
        if cache_key in self.cache_turnover:
            return self.cache_turnover[cache_key]
            
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        defending_team = away_team if current_team == home_team else home_team
        def_rate = team_defensive_profiles.get(defending_team, {}).get(current_zone, 0.0)
        
        cx = int(current_zone.split('_')[1])
        cy = int(current_zone.split('_')[2])
        nx = int(next_zone.split('_')[1])
        ny = int(next_zone.split('_')[2])
        dx = (nx - cx) * 20
        dy = (ny - cy) * 16
        pass_length = np.sqrt(dx**2 + dy**2)
        pass_angle = np.arctan2(dy, dx)
        
        feats = self._compile_features(current_zone, player_on_ball, player_profiles, 
                                       {}, {}, player_to_team, 
                                       home_team, away_team, history, score_differential, possession_duration,
                                       pass_sequence_index, pass_length, pass_angle, possession_directions,
                                       time_ratio, under_pressure, next_zone)
        feats['opp_defensive_rate'] = def_rate
        
        df_feats = pd.DataFrame([feats])
        dest_features = [
            'start_zone_x', 'start_zone_y',
            'zone_emb_0', 'zone_emb_1', 'zone_emb_2', 'zone_emb_3',
            'player_emb_0', 'player_emb_1', 'player_emb_2', 'player_emb_3',
            'player_emb_4', 'player_emb_5', 'player_emb_6', 'player_emb_7',
            'opp_defensive_rate', 'opp_gk_save_ratio',
            'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
            'score_differential', 'possession_duration', 'pass_sequence_index',
            'player_role',
            'prev_pass_direction_1', 'prev_pass_direction_2', 'prev_pass_direction_3',
            'under_pressure', 'game_state_momentum',
            'prev_1_zone_emb_0', 'prev_1_zone_emb_1', 'prev_1_zone_emb_2', 'prev_1_zone_emb_3', 'prev_1_success',
            'prev_2_zone_emb_0', 'prev_2_zone_emb_1', 'prev_2_zone_emb_2', 'prev_2_zone_emb_3', 'prev_2_success'
        ] + [f'target_def_density_{tx}_{ty}' for tx in range(6) for ty in range(5)]
        
        outcome_features = dest_features + ['pass_length', 'pass_angle', 'pressure_differential']
        df_feats = df_feats[outcome_features]
        
        if isinstance(self.outcome_model, torch.nn.Module):
            with torch.no_grad():
                continuous_cols = [c for c in df_feats.columns if c not in ['player_role', 'start_zone_x', 'start_zone_y']]
                df_feats_scaled = df_feats.copy()
                df_feats_scaled[continuous_cols] = self.outcome_scaler.transform(df_feats[continuous_cols])
                feats_tensor = torch.FloatTensor(df_feats_scaled.values)
                turnover_prob = self.outcome_model(feats_tensor).item()
        else:
            outcome_probs = self.outcome_model.predict_proba(df_feats)[0]
            turnover_prob = outcome_probs[1] if len(outcome_probs) > 1 else 0.12
        
        self.cache_turnover[cache_key] = turnover_prob
        return turnover_prob
