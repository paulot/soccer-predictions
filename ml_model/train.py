import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

# Try importing xgboost, fallback to RandomForest if not installed or fails to load
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def train_models(model_type='random_forest', mode='iteration'):
    print(f"Training models using architecture: {model_type.upper()} (Mode: {mode.upper()})...")
    
    # 1. Load dataset
    csv_path = f"data/ml_training_data_{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"Training data {csv_path} not found. Running features extraction first...")
        from ml_model.features import extract_features_and_targets
        extract_features_and_targets(mode)
        
    df = pd.read_csv(csv_path)
    
    # Define features
    dest_features = [
        'start_zone_x', 'start_zone_y', 'passer_accuracy', 'passer_progressive_ratio',
        'opp_defensive_rate', 'opp_gk_save_ratio', 'manager_directness', 'manager_width',
        'score_differential', 'possession_duration', 'pass_sequence_index',
        'prev_1_zone_x', 'prev_1_zone_y', 'prev_1_success',
        'prev_2_zone_x', 'prev_2_zone_y', 'prev_2_success'
    ]
    
    outcome_features = dest_features + ['pass_length', 'pass_angle']
    
    X_outcome = df[outcome_features]
    X_dest = df[dest_features]
    y_outcome = df['outcome'] # 0 = Success, 1 = Turnover
    
    # Flatten destination zone into a single class (0 to 29)
    y_dest = df['end_zone_x'] * 5 + df['end_zone_y']
    
    # --- MODEL 1: OUTCOME MODEL (Success vs Turnover) ---
    print("Training Outcome Model...")
    X_train, X_val, y_train, y_val = train_test_split(X_outcome, y_outcome, test_size=0.2, random_state=42)
    
    if model_type == 'logistic_regression':
        outcome_model = LogisticRegression(max_iter=1000)
    elif model_type == 'xgboost' and HAS_XGB:
        outcome_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
    else: # Default: random_forest
        if model_type == 'xgboost':
            print("  XGBoost not installed or fails to load. Falling back to Random Forest.")
        outcome_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        
    outcome_model.fit(X_train, y_train)
    val_preds = outcome_model.predict(X_val)
    val_probs = outcome_model.predict_proba(X_val)
    print(f"  Outcome Model Accuracy: {accuracy_score(y_val, val_preds):.2%}")
    print(f"  Outcome Model Log Loss: {log_loss(y_val, val_probs):.4f}")
    
    # --- MODEL 2: DESTINATION MODEL (Where does a successful pass go?) ---
    print("Training Destination Model...")
    # Filter for successful passes only
    success_mask = (y_outcome == 0)
    X_success = X_dest[success_mask]
    y_dest_success = y_dest[success_mask]
    
    X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_success, y_dest_success, test_size=0.2, random_state=42)
    
    if model_type == 'logistic_regression':
        dest_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    elif model_type == 'xgboost' and HAS_XGB:
        dest_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
    else: # Default: random_forest
        dest_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        
    dest_model.fit(X_train_d, y_train_d)
    val_preds_d = dest_model.predict(X_val_d)
    val_probs_d = dest_model.predict_proba(X_val_d)
    print(f"  Destination Model Accuracy: {accuracy_score(y_val_d, val_preds_d):.2%}")
    print(f"  Destination Model Log Loss: {log_loss(y_val_d, val_probs_d):.4f}")
    
    # Save models
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)
    
    outcome_path = os.path.join(model_dir, f"{model_type}_{mode}_outcome.pkl")
    dest_path = os.path.join(model_dir, f"{model_type}_{mode}_destination.pkl")
    
    with open(outcome_path, 'wb') as f:
        pickle.dump(outcome_model, f)
    with open(dest_path, 'wb') as f:
        pickle.dump(dest_model, f)
        
    print(f"Saved models to {outcome_path} and {dest_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='random_forest', choices=['logistic_regression', 'random_forest', 'xgboost'])
    parser.add_argument('--mode', type=str, default='iteration', choices=['iteration', 'production'])
    args = parser.parse_args()
    
    train_models(args.model, args.mode)
