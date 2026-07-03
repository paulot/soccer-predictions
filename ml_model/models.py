import os
import pickle
import numpy as np
import pandas as pd

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
        # Caches to avoid expensive predict_proba calls in the MCMC loop
        self.cache_probs = {}
        self.cache_turnover = {}
            
    def _compile_features(self, current_zone, player_on_ball, player_profiles, 
                          manager_profiles, team_to_manager, player_to_team, 
                          home_team, away_team, history=None, score_differential=0, possession_duration=0.0,
                          pass_sequence_index=0, pass_length=0.0, pass_angle=0.0):
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        
        start_x = int(current_zone.split('_')[1])
        start_y = int(current_zone.split('_')[2])
        
        p_profile = player_profiles.get(player_on_ball, {'accuracy': 0.80, 'progressive_ratio': 0.25})
        passer_acc = p_profile.get('accuracy', 0.80)
        passer_prog = p_profile.get('progressive_ratio', 0.25)
        
        if history is None:
            history = [(-1, -1, -1), (-1, -1, -1)]
        p1, p2 = history[0], history[1]
        
        return {
            'start_zone_x': start_x,
            'start_zone_y': start_y,
            'passer_accuracy': passer_acc,
            'passer_progressive_ratio': passer_prog,
            'opp_defensive_rate': 0.0,
            'opp_gk_save_ratio': 0.70,
            'manager_directness': 5,
            'manager_width': 5,
            'score_differential': score_differential,
            'possession_duration': possession_duration,
            'pass_sequence_index': pass_sequence_index,
            'pass_length': pass_length,
            'pass_angle': pass_angle,
            'prev_1_zone_x': p1[0],
            'prev_1_zone_y': p1[1],
            'prev_1_success': p1[2],
            'prev_2_zone_x': p2[0],
            'prev_2_zone_y': p2[1],
            'prev_2_success': p2[2]
        }

    def get_transition_probabilities(self, current_zone, player_on_ball, player_profiles, 
                                      manager_profiles, team_to_manager, player_to_team, 
                                      home_team, away_team, zones, history=None, score_differential=0, possession_duration=0.0,
                                      pass_sequence_index=0):
        if history is None:
            history = [(-1, -1, -1), (-1, -1, -1)]
        hist_tuple = tuple(history)
        cache_key = (player_on_ball, current_zone, hist_tuple, score_differential, round(possession_duration, 1), pass_sequence_index)
        if cache_key in self.cache_probs:
            return self.cache_probs[cache_key]
            
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        mgr_name = team_to_manager.get(current_team)
        mgr = manager_profiles.get(mgr_name, {"directness": 5, "width": 5})
        
        # Build features (pass_length and pass_angle are omitted for destination model)
        feats = self._compile_features(current_zone, player_on_ball, player_profiles, 
                                       manager_profiles, team_to_manager, player_to_team, 
                                       home_team, away_team, history, score_differential, possession_duration,
                                       pass_sequence_index)
        feats['manager_directness'] = mgr.get('directness', 5)
        feats['manager_width'] = mgr.get('width', 5)
        
        df_feats = pd.DataFrame([feats])
        feature_cols = [
            'start_zone_x', 'start_zone_y', 'passer_accuracy', 'passer_progressive_ratio',
            'opp_defensive_rate', 'opp_gk_save_ratio', 'manager_directness', 'manager_width',
            'score_differential', 'possession_duration', 'pass_sequence_index',
            'prev_1_zone_x', 'prev_1_zone_y', 'prev_1_success',
            'prev_2_zone_x', 'prev_2_zone_y', 'prev_2_success'
        ]
        df_feats = df_feats[feature_cols]
        
        dest_probs = self.destination_model.predict_proba(df_feats)[0]
        
        zone_probs = pd.Series(0.0, index=zones)
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
                                  pass_sequence_index=0, next_zone=None):
        # Fallback if next_zone is not provided
        if next_zone is None:
            next_zone = current_zone
            
        if history is None:
            history = [(-1, -1, -1), (-1, -1, -1)]
        hist_tuple = tuple(history)
        cache_key = (player_on_ball, current_zone, next_zone, hist_tuple, score_differential, round(possession_duration, 1), pass_sequence_index)
        if cache_key in self.cache_turnover:
            return self.cache_turnover[cache_key]
            
        current_team = player_to_team.get(player_on_ball) if player_on_ball else np.random.choice([home_team, away_team])
        defending_team = away_team if current_team == home_team else home_team
        def_rate = team_defensive_profiles.get(defending_team, {}).get(current_zone, 0.0)
        
        # Calculate simulated pass length and angle
        cx = int(current_zone.split('_')[1])
        cy = int(current_zone.split('_')[2])
        nx = int(next_zone.split('_')[1])
        ny = int(next_zone.split('_')[2])
        dx = (nx - cx) * 20
        dy = (ny - cy) * 16
        pass_length = np.sqrt(dx**2 + dy**2)
        pass_angle = np.arctan2(dy, dx)
        
        # Build features
        feats = self._compile_features(current_zone, player_on_ball, player_profiles, 
                                       {}, {}, player_to_team, 
                                       home_team, away_team, history, score_differential, possession_duration,
                                       pass_sequence_index, pass_length, pass_angle)
        feats['opp_defensive_rate'] = def_rate
        
        df_feats = pd.DataFrame([feats])
        feature_cols = [
            'start_zone_x', 'start_zone_y', 'passer_accuracy', 'passer_progressive_ratio',
            'opp_defensive_rate', 'opp_gk_save_ratio', 'manager_directness', 'manager_width',
            'score_differential', 'possession_duration', 'pass_sequence_index',
            'prev_1_zone_x', 'prev_1_zone_y', 'prev_1_success',
            'prev_2_zone_x', 'prev_2_zone_y', 'prev_2_success',
            'pass_length', 'pass_angle'
        ]
        df_feats = df_feats[feature_cols]
        
        outcome_probs = self.outcome_model.predict_proba(df_feats)[0]
        turnover_prob = outcome_probs[1] if len(outcome_probs) > 1 else 0.12
        
        self.cache_turnover[cache_key] = turnover_prob
        return turnover_prob
