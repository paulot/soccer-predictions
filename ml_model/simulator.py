import numpy as np
import pandas as pd
from mcmc_simulation import get_zone_players

def simulate_full_match(home_team, away_team, transition_model, df_events, player_profiles, gk_profiles, 
                        team_defensive_profiles, manager_profiles, team_to_manager, player_to_team, 
                        zones, num_possessions=100):
    """
    Simulates a single football match using a modular TransitionModel (Heuristic or ML).
    """
    home_goals = 0
    away_goals = 0
    
    # Identify goalkeepers for both teams
    home_gk = None
    away_gk = None
    for player, team in player_to_team.items():
        if player in gk_profiles:
            if team == home_team:
                home_gk = player
            elif team == away_team:
                away_gk = player
                
    # Start with a kickoff (stochastically given to one team)
    current_team = np.random.choice([home_team, away_team])
    current_zone = "Z_2_2" # Start in central midfield
    
    for _ in range(num_possessions):
        chain_active = True
        # Initialize sequence history buffer for the possession: [prev_1, prev_2]
        # Each entry is (zone_x, zone_y, success)
        history = [(-1, -1, -1), (-1, -1, -1)]
        possession_duration = 0.0
        
        while chain_active:
            # Get players in this zone
            zone_players = get_zone_players(df_events, current_zone)
            
            # Filter for players who actually play for the team currently in possession
            team_zone_players = {p: w for p, w in zone_players.items() if player_to_team.get(p) == current_team}
            
            if not team_zone_players:
                player_on_ball = None
            else:
                # Re-normalize weights
                total_w = sum(team_zone_players.values())
                team_zone_players = {p: w/total_w for p, w in team_zone_players.items()}
                player_on_ball = np.random.choice(list(team_zone_players.keys()), p=list(team_zone_players.values()))
            
            # Calculate score differential from the perspective of the team in possession
            score_differential = home_goals - away_goals if current_team == home_team else away_goals - home_goals
            
            # Get transition probabilities from the modular model
            zone_probs = transition_model.get_transition_probabilities(
                current_zone, player_on_ball, player_profiles, 
                manager_profiles, team_to_manager, player_to_team, 
                home_team, away_team, zones, history=history,
                score_differential=score_differential, possession_duration=possession_duration
            )
            
            if zone_probs.sum() == 0:
                # Fallback: turnover
                current_team = away_team if current_team == home_team else home_team
                break
                
            # Sample next zone
            next_zone = np.random.choice(zone_probs.index, p=zone_probs.values)
            
            # --- TRANSITION LOGIC ---
            # 1. Shot Opportunity (Entering Z_5_x)
            if next_zone.startswith("Z_5_"):
                # Determine shooter's conversion rate (fallback to 10%)
                if player_on_ball and player_on_ball in player_profiles:
                    conversion = player_profiles[player_on_ball].get('shot_conversion', 0.10)
                    if conversion == 0.0:
                        conversion = 0.10
                else:
                    conversion = 0.10
                    
                # Determine opposing goalkeeper's save multiplier
                opp_gk = away_gk if current_team == home_team else home_gk
                save_ratio = gk_profiles.get(opp_gk, 0.70)
                gk_multiplier = max(0.5, min(1.5, (1.0 - save_ratio) / 0.30))
                
                final_conversion_rate = conversion * gk_multiplier
                
                if np.random.rand() < final_conversion_rate:
                    if current_team == home_team:
                        home_goals += 1
                    else:
                        away_goals += 1
                
                # Goal ends possession, opponent kicks off from center
                current_team = away_team if current_team == home_team else home_team
                current_zone = "Z_2_2"
                chain_active = False
                
            # 2. Turnover (dynamically determined by the modular model)
            else:
                turnover_prob = transition_model.get_turnover_probability(
                    current_zone, player_on_ball, player_profiles, 
                    team_defensive_profiles, player_to_team, home_team, away_team, history=history,
                    score_differential=score_differential, possession_duration=possession_duration
                )
                
                if np.random.rand() < turnover_prob: 
                    current_team = away_team if current_team == home_team else home_team
                    current_zone = next_zone
                    chain_active = False
                else:
                    # Update history buffer with this successful pass
                    curr_x = int(current_zone.split('_')[1])
                    curr_y = int(current_zone.split('_')[2])
                    history = [(curr_x, curr_y, 1), history[0]]
                    possession_duration += 3.0
                    
                    current_zone = next_zone
                    
    return home_goals, away_goals
