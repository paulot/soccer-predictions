import os
import pickle
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from typing import List, Any, Optional

# xgboost will be imported lazily to avoid OpenMP conflicts with PyTorch.


def train_models(model_type: str = "random_forest", mode: str = "iteration") -> None:
    print(f"Training models using architecture: {model_type.upper()} (Mode: {mode.upper()})...")

    # 1. Load dataset
    csv_path: str = f"data/ml_training_data_{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"Training data {csv_path} not found. Running features extraction first...")
        from ml_model.features import extract_features_and_targets

        extract_features_and_targets(mode)

    df: pd.DataFrame = pd.read_csv(csv_path)

    # Define features leveraging embeddings and new contextual features
    dest_features: List[str] = [
        "start_zone_x",
        "start_zone_y",
        "zone_emb_0",
        "zone_emb_1",
        "zone_emb_2",
        "zone_emb_3",
        "player_emb_0",
        "player_emb_1",
        "player_emb_2",
        "player_emb_3",
        "player_emb_4",
        "player_emb_5",
        "player_emb_6",
        "player_emb_7",
        "opp_defensive_rate",
        "opp_gk_save_ratio",
        "manager_emb_0",
        "manager_emb_1",
        "manager_emb_2",
        "manager_emb_3",
        "score_differential",
        "possession_duration",
        "pass_sequence_index",
        "player_role",
        "prev_pass_direction_1",
        "prev_pass_direction_2",
        "prev_pass_direction_3",
        "under_pressure",
        "game_state_momentum",
        "prev_1_zone_emb_0",
        "prev_1_zone_emb_1",
        "prev_1_zone_emb_2",
        "prev_1_zone_emb_3",
        "prev_1_success",
        "prev_2_zone_emb_0",
        "prev_2_zone_emb_1",
        "prev_2_zone_emb_2",
        "prev_2_zone_emb_3",
        "prev_2_success",
    ] + [f"target_def_density_{tx}_{ty}" for tx in range(6) for ty in range(5)]

    outcome_features: List[str] = dest_features + ["pass_length", "pass_angle", "pressure_differential"]

    X_outcome: pd.DataFrame = df[outcome_features]
    X_dest: pd.DataFrame = df[dest_features]
    y_outcome: pd.Series = df["outcome"]  # 0 = Success, 1 = Turnover

    # Flatten destination zone into a single class (0 to 29)
    y_dest: pd.Series = df["end_zone_x"] * 5 + df["end_zone_y"]

    # --- MODEL 1: OUTCOME MODEL (Success vs Turnover) ---
    print("Training Outcome Model...")
    X_train, X_val, y_train, y_val = train_test_split(X_outcome, y_outcome, test_size=0.2, random_state=42)

    outcome_model: Any = None
    outcome_scaler: Optional[Any] = None

    if model_type == "logistic_regression":
        outcome_model = LogisticRegression(max_iter=1000)
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            outcome_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
        except ImportError:
            print("  XGBoost not installed. Falling back to Random Forest.")
            outcome_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    elif model_type == "neural_network":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import StandardScaler
        from ml_model.pytorch_models import OutcomeNN

        # Fit and apply scaler only to continuous features (exclude player_role and start_zone_x/y)
        continuous_cols: List[str] = [
            c for c in X_train.columns if c not in ["player_role", "start_zone_x", "start_zone_y"]
        ]
        outcome_scaler = StandardScaler()

        X_train_scaled = X_train.copy()
        X_train_scaled[continuous_cols] = outcome_scaler.fit_transform(X_train[continuous_cols])
        X_val_scaled = X_val.copy()
        X_val_scaled[continuous_cols] = outcome_scaler.transform(X_val[continuous_cols])

        X_train_t: torch.Tensor = torch.FloatTensor(X_train_scaled.values)
        y_train_t: torch.Tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
        X_val_t: torch.Tensor = torch.FloatTensor(X_val_scaled.values)

        outcome_model = OutcomeNN(X_train.shape[1], role_idx=23)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(outcome_model.parameters(), lr=0.005, weight_decay=1e-4)

        outcome_model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = outcome_model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch} | Loss: {loss.item():.4f}", flush=True)

        outcome_model.eval()
    else:  # Default: random_forest
        outcome_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

    if model_type == "neural_network":
        with torch.no_grad():
            val_probs = outcome_model(X_val_t).numpy()
            val_probs_2d = np.hstack([1.0 - val_probs, val_probs])
            val_preds = (val_probs > 0.5).astype(int).flatten()
        print(f"  Outcome Model Accuracy: {accuracy_score(y_val, val_preds):.2%}")
        print(f"  Outcome Model Log Loss: {log_loss(y_val, val_probs_2d):.4f}")
    else:
        outcome_model.fit(X_train, y_train)
        val_preds = outcome_model.predict(X_val)
        val_probs = outcome_model.predict_proba(X_val)
        print(f"  Outcome Model Accuracy: {accuracy_score(y_val, val_preds):.2%}")
        print(f"  Outcome Model Log Loss: {log_loss(y_val, val_probs):.4f}")

    # --- MODEL 2: DESTINATION MODEL (Where does a successful pass go?) ---
    print("Training Destination Model...")
    success_mask = y_outcome == 0
    X_success = X_dest[success_mask]
    y_dest_success = y_dest[success_mask]

    X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_success, y_dest_success, test_size=0.2, random_state=42)

    dest_model: Any = None
    dest_scaler: Optional[Any] = None

    if model_type == "logistic_regression":
        dest_model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier

            dest_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
        except ImportError:
            print("  XGBoost not installed. Falling back to Random Forest.")
            dest_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    elif model_type == "neural_network":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.preprocessing import StandardScaler
        from ml_model.pytorch_models import DestinationNN

        # Fit and apply scaler only to continuous features (exclude player_role and start_zone_x/y)
        continuous_cols_d: List[str] = [
            c for c in X_train_d.columns if c not in ["player_role", "start_zone_x", "start_zone_y"]
        ]
        dest_scaler = StandardScaler()

        X_train_d_scaled = X_train_d.copy()
        X_train_d_scaled[continuous_cols_d] = dest_scaler.fit_transform(X_train_d[continuous_cols_d])
        X_val_d_scaled = X_val_d.copy()
        X_val_d_scaled[continuous_cols_d] = dest_scaler.transform(X_val_d[continuous_cols_d])

        X_train_t_d: torch.Tensor = torch.FloatTensor(X_train_d_scaled.values)
        y_train_t_d: torch.Tensor = torch.LongTensor(y_train_d.values)
        X_val_t_d: torch.Tensor = torch.FloatTensor(X_val_d_scaled.values)

        dest_model = DestinationNN(X_train_d.shape[1], role_idx=23, def_density_start_idx=39, output_dim=30)
        criterion_d = nn.CrossEntropyLoss()
        optimizer_d = optim.Adam(dest_model.parameters(), lr=0.005, weight_decay=1e-4)

        dest_model.train()
        for epoch in range(150):
            optimizer_d.zero_grad()
            outputs_d = dest_model(X_train_t_d)
            loss_d = criterion_d(outputs_d, y_train_t_d)
            loss_d.backward()
            optimizer_d.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch} | Loss: {loss_d.item():.4f}", flush=True)

        dest_model.eval()
    else:  # Default: random_forest
        dest_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)

    if model_type == "neural_network":
        with torch.no_grad():
            logits = dest_model(X_val_t_d)
            val_probs_d = torch.softmax(logits, dim=1).numpy()
            val_preds_d = val_probs_d.argmax(axis=1)
        print(f"  Destination Model Accuracy: {accuracy_score(y_val_d, val_preds_d):.2%}")
        print(f"  Destination Model Log Loss: {log_loss(y_val_d, val_probs_d, labels=list(range(30))):.4f}")
    else:
        dest_model.fit(X_train_d, y_train_d)
        val_preds_d = dest_model.predict(X_val_d)
        val_probs_d = dest_model.predict_proba(X_val_d)
        print(f"  Destination Model Accuracy: {accuracy_score(y_val_d, val_preds_d):.2%}")
        print(f"  Destination Model Log Loss: {log_loss(y_val_d, val_probs_d):.4f}")

    # Save models and scalers
    model_dir: str = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    if model_type == "neural_network":
        outcome_model.eval()
        dest_model.eval()
        outcome_model = torch.jit.script(outcome_model)
        dest_model = torch.jit.script(dest_model)

    outcome_path: str = os.path.join(model_dir, f"{model_type}_{mode}_outcome.pkl")
    dest_path: str = os.path.join(model_dir, f"{model_type}_{mode}_destination.pkl")

    if model_type == "neural_network":
        torch.jit.save(outcome_model, outcome_path)
        torch.jit.save(dest_model, dest_path)
    else:
        with open(outcome_path, "wb") as f:
            pickle.dump(outcome_model, f)
        with open(dest_path, "wb") as f:
            pickle.dump(dest_model, f)

    if model_type == "neural_network":
        with open(os.path.join(model_dir, "neural_network_outcome_scaler.pkl"), "wb") as f:
            pickle.dump(outcome_scaler, f)
        with open(os.path.join(model_dir, "neural_network_destination_scaler.pkl"), "wb") as f:
            pickle.dump(dest_scaler, f)
        print("Saved PyTorch scalers to data/models/")

    print(f"Saved models to {outcome_path} and {dest_path}")


@click.command(help="Train transition and outcome models.")
@click.option(
    "--model",
    type=click.Choice(["logistic_regression", "random_forest", "xgboost", "neural_network"]),
    default="random_forest",
    help="Classifier architecture to train",
)
@click.option(
    "--mode", type=click.Choice(["iteration", "production"]), default="iteration", help="Mode: iteration or production"
)
def main(model: str, mode: str) -> None:
    train_models(model, mode)


if __name__ == "__main__":
    main()
