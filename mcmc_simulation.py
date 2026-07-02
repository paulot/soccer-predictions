import numpy as np
import pandas as pd
try:
    from statsbombpy import sb
except ImportError:
    print("Please install statsbombpy first: pip install statsbombpy")
    exit(1)

def build_30_zone_grid():
    """Creates a 6x5 grid representing the 30 zones on a pitch."""
    zones = []
    for x in range(6):
        for y in range(5):
            zones.append(f"Z_{x}_{y}")
    return zones

def map_coordinates_to_zone(x, y):
    """
    Maps StatsBomb coordinates to a 30-zone grid (6x5).
    StatsBomb pitch dimensions: X: 0-120, Y: 0-80
    """
    # X divides into 6 zones of 20 length
    zone_x = min(int(x / 20), 5)
    # Y divides into 5 zones of 16 width
    zone_y = min(int(y / 16), 4)
    return f"Z_{zone_x}_{zone_y}"

def fetch_real_statsbomb_data():
    """
    Fetches real event data from StatsBomb Open Data.
    For this example, we use the 2022 World Cup Final: Argentina vs France (Match ID: 3869685)
    """
    print("Fetching 2022 World Cup Final (Argentina vs France) events from StatsBomb...")
    events = sb.events(match_id=3869685)
    
    # Filter for passes to build our spatial transition matrix
    passes = events[events['type'] == 'Pass'].copy()
    passes = passes.dropna(subset=['location', 'pass_end_location'])
    
    # Map raw coordinates to our discrete zones
    passes['start_zone'] = passes['location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
    passes['end_zone'] = passes['pass_end_location'].apply(lambda loc: map_coordinates_to_zone(loc[0], loc[1]))
    
    # In StatsBomb, a null 'pass_outcome' means the pass was successful
    passes['event_type'] = passes['pass_outcome'].apply(lambda outcome: 'Turnover' if pd.notnull(outcome) else 'Pass')
    
    return passes

def calculate_player_profiles(df_events):
    """
    Calculates player-specific metrics from the event data:
    - progressive_ratio: proportion of successful passes that move the ball forward (increase in X)
    - pass_accuracy: ratio of successful passes to total passes
    """
    profiles = {}
    players = df_events['player'].dropna().unique()
    
    for player in players:
        player_passes = df_events[df_events['player'] == player]
        total_attempts = len(player_passes)
        if total_attempts < 5: # Skip players with too few passes
            continue
            
        successful = player_passes[player_passes['event_type'] == 'Pass']
        accuracy = len(successful) / total_attempts
        
        # Calculate progressive ratio (passes moving the ball forward on the X-axis)
        forward_passes = 0
        for _, row in successful.iterrows():
            start_x = int(row['start_zone'].split('_')[1])
            end_x = int(row['end_zone'].split('_')[1])
            if end_x > start_x:
                forward_passes += 1
                
        prog_ratio = forward_passes / len(successful) if len(successful) > 0 else 0
        
        profiles[player] = {
            "accuracy": accuracy,
            "progressive_ratio": prog_ratio,
            "total_passes": total_attempts
        }
        
    return profiles

def get_zone_players(df_events, zone):
    """Returns players who historically made passes from this zone, with their frequencies."""
    zone_passes = df_events[df_events['start_zone'] == zone]
    if zone_passes.empty:
        return {}
    return zone_passes['player'].value_counts(normalize=True).to_dict()

def build_baseline_transition_matrix(df_events, zones):
    """
    Builds the baseline transition matrix P(End_Zone | Start_Zone) using real event frequencies.
    """
    matrix = pd.DataFrame(0.0, index=zones, columns=zones)
    
    # Count successful passes between zones
    successful_passes = df_events[df_events['event_type'] == 'Pass']
    transitions = successful_passes.groupby(['start_zone', 'end_zone']).size().reset_index(name='count')
    
    for _, row in transitions.iterrows():
        matrix.at[row['start_zone'], row['end_zone']] = row['count']
        
    # Row normalize so probabilities sum to 1.0
    row_sums = matrix.sum(axis=1)
    matrix = matrix.div(row_sums, axis=0)
    matrix = matrix.fillna(0)
    
    return matrix

def apply_player_modifier(row_probs, player_profile, start_zone, zones):
    """
    Adjusts the transition probabilities of a zone based on the player's progressive passing profile.
    """
    start_x = int(start_zone.split('_')[1])
    
    # Scale forward passes based on progressive_ratio (baseline average is ~0.33)
    prog_multiplier = player_profile['progressive_ratio'] / 0.33
    
    modified_probs = row_probs.copy()
    for zone in zones:
        end_x = int(zone.split('_')[1])
        if end_x > start_x:
            modified_probs[zone] *= prog_multiplier
            
    # Re-normalize the probabilities
    if modified_probs.sum() > 0:
        modified_probs = modified_probs / modified_probs.sum()
        
    return modified_probs

def simulate_mcmc_possession_chain(start_zone, base_matrix, df_events, player_profiles, zones, max_steps=10):
    """
    Simulates a possession chain, dynamically selecting the player on the ball 
    based on the zone and applying their specific pass modifiers.
    """
    current_zone = start_zone
    chain = [current_zone]
    
    print(f"\n--- Simulating Player-Modified Possession Chain starting in {start_zone} ---")
    for step in range(max_steps):
        # 1. Determine who has the ball in this zone based on historical match data
        zone_players = get_zone_players(df_events, current_zone)
        if not zone_players:
            print(f"Step {step+1}: Chain ended in {current_zone} (No player data for this zone)")
            break
            
        # Select player stochastically based on who played in this zone
        player_on_ball = np.random.choice(list(zone_players.keys()), p=list(zone_players.values()))
        
        # 2. Get baseline probabilities for this zone
        zone_probs = base_matrix.loc[current_zone].copy()
        if zone_probs.sum() == 0:
            print(f"Step {step+1}: Chain ended in {current_zone} (No transition data)")
            break
            
        # 3. Apply player modifier if they have a profile
        if player_on_ball in player_profiles:
            profile = player_profiles[player_on_ball]
            zone_probs = apply_player_modifier(zone_probs, profile, current_zone, zones)
            player_info = f"[{player_on_ball} | Acc: {profile['accuracy']:.2f}, Prog: {profile['progressive_ratio']:.2f}]"
        else:
            player_info = f"[{player_on_ball} (No Profile)]"
            
        # 4. Sample the next zone
        next_zone = np.random.choice(base_matrix.columns, p=zone_probs.values)
        chain.append(next_zone)
        
        print(f"Step {step+1}: {player_info} in {current_zone} -> passes to {next_zone}")
        current_zone = next_zone
        
        # End if ball enters the box
        if current_zone.startswith("Z_5_"):
            print("Outcome: Ball entered the attacking penalty box! (Shot opportunity)")
            break
            
    return chain

if __name__ == "__main__":
    zones = build_30_zone_grid()
    
    # 1. Fetch real match data
    try:
        df = fetch_real_statsbomb_data()
        print(f"Successfully loaded {len(df)} passes from StatsBomb.")
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        exit(1)
        
    # 2. Build Profiles and Matrices
    player_profiles = calculate_player_profiles(df)
    matrix = build_baseline_transition_matrix(df, zones)
    
    # Print a few top player profiles for demonstration
    print("\nSample Player Profiles (World Cup Final):")
    for player in ["Lionel Andrés Messi Cuccittini", "Enzo Jeremías Fernández", "Antoine Griezmann", "Aurélien Djani Tchouaméni"]:
        if player in player_profiles:
            prof = player_profiles[player]
            print(f"- {player}: Accuracy={prof['accuracy']:.2f}, Progressive Ratio={prof['progressive_ratio']:.2f} (Total Passes: {prof['total_passes']})")
            
    # 3. Run MCMC Simulations with Player Modifiers
    simulate_mcmc_possession_chain("Z_2_2", matrix, df, player_profiles, zones)
    simulate_mcmc_possession_chain("Z_3_2", matrix, df, player_profiles, zones)
