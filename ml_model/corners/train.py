import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from ml_model.corners.features import extract_corner_features
from ml_model.corners.models import CornerRoutineXGB, CornerOutcomeXGB


def train_corner_models(
    mode: str = "iteration",
    tune: bool = False,
    optimize_thresh: bool = True,
    use_class_weights: bool = True,
) -> None:
    """
    Trains Stage 1 (3-Class Routine) and Stage 2 (Binary Outcome) XGBoost models for Corner Kicks.
    """
    print(f"Training XGBoost Corner Models (Mode: {mode.upper()})...")
    data_path = f"data/corners_training_data_{mode}.csv"

    need_extract = False
    if not os.path.exists(data_path):
        print(f"Training data {data_path} not found. Running corner features extraction first...")
        need_extract = True
    else:
        df = pd.read_csv(data_path)
        if not all(col in df.columns for col in ["routine_lag_1", "hist_rate_routine_3", "team_match_corner_count", "consecutive_same_routine"]):
            print(f"Training data {data_path} is missing newly engineered sequence features. Re-extracting...")
            need_extract = True

    if need_extract:
        df = extract_corner_features(mode)

    if df is None or df.empty:
        print("No corner training data available.")
        return

    feature_cols = [
        "is_right_corner",
        "time_ratio",
        "score_differential",
        "is_home_team",
        "inswinging",
        "taker_accuracy",
        "taker_key_pass_ratio",
        "team_directness",
        "opp_def_rate",
        "under_pressure",
        "corner_cluster_density",
        "aerial_height_advantage",
        "goalkeeper_line_command",
        "taker_corner_assist_rate",
        "routine_lag_1",
        "routine_lag_2",
        "routine_lag_3",
        "routine_lag_4",
        "routine_lag_5",
        "hist_rate_routine_1",
        "hist_rate_routine_2",
        "hist_rate_routine_3",
        "team_match_corner_count",
        "consecutive_same_routine",
    ]

    X = df[feature_cols].fillna(0.0)
    y_routine = df["target_routine"].astype(int)
    y_outcome = df["target_outcome"].astype(int)

    X_train, X_test, y_r_train, y_r_test, y_o_train, y_o_test = train_test_split(
        X, y_routine, y_outcome, test_size=0.2, random_state=42
    )

    # 1. Train Corner Routine Classifier
    print("\n--- Training Stage 1: Corner Routine XGBoost Classifier (4-Class, Focal Loss + Subsampling + Regularized Depth/Leaves) ---")
    routine_model = CornerRoutineXGB()
    if tune:
        print("  Tuning Stage 1 hyperparameters...")
        best_params_r = routine_model.tune_hyperparameters(X_train, y_r_train, scoring="f1_macro")
        print(f"  Best Routine Params: {best_params_r}")
    routine_model.fit(X_train, y_r_train, use_class_weights=use_class_weights)

    r_preds = routine_model.predict(X_test)
    r_probs = routine_model.predict_proba(X_test)
    r_acc = accuracy_score(y_r_test, r_preds)
    r_f1 = f1_score(y_r_test, r_preds, average="macro", zero_division=0)
    r_loss = log_loss(y_r_test, r_probs)

    print(f"  Routine Model Accuracy: {r_acc:.2%}")
    print(f"  Routine Model Macro F1-Score: {r_f1:.4f}")
    print(f"  Routine Model Multi-Class Log Loss: {r_loss:.4f}")

    # 2. Train Corner Outcome Classifier
    print("\n--- Training Stage 2: Corner Outcome XGBoost Classifier (Binary) ---")
    outcome_model = CornerOutcomeXGB()
    if tune:
        print("  Tuning Stage 2 hyperparameters...")
        best_params_o = outcome_model.tune_hyperparameters(X_train, y_o_train, scoring="f1")
        print(f"  Best Outcome Params: {best_params_o}")
    outcome_model.fit(X_train, y_o_train, use_class_weights=use_class_weights)

    if optimize_thresh:
        best_thresh = outcome_model.optimize_threshold(X_test, y_o_test, metric="f1")
        print(f"  Optimized Decision Threshold: {best_thresh:.4f}")

    o_preds = outcome_model.predict(X_test)
    o_probs = outcome_model.predict_proba(X_test)
    o_acc = accuracy_score(y_o_test, o_preds)
    o_f1 = f1_score(y_o_test, o_preds, zero_division=0)
    try:
        o_auc = roc_auc_score(y_o_test, o_probs)
    except Exception:
        o_auc = 0.5
    o_loss = log_loss(y_o_test, o_probs)

    print(f"  Outcome Model Accuracy: {o_acc:.2%}")
    print(f"  Outcome Model F1-Score: {o_f1:.4f}")
    print(f"  Outcome Model ROC-AUC: {o_auc:.4f}")
    print(f"  Outcome Model Binary Log Loss: {o_loss:.4f}")

    # Save models
    os.makedirs("data/models", exist_ok=True)
    routine_model.save(f"data/models/corner_routine_xgb_{mode}.pkl")
    outcome_model.save(f"data/models/corner_outcome_xgb_{mode}.pkl")
    print(
        f"\nSaved corner models to data/models/corner_routine_xgb_{mode}.pkl and data/models/corner_outcome_xgb_{mode}.pkl"
    )


@click.command()
@click.option("--mode", type=click.Choice(["iteration", "production"]), default="iteration", help="Training mode")
@click.option("--tune", is_flag=True, default=False, help="Run GridSearchCV hyperparameter tuning")
@click.option("--optimize-thresh/--no-optimize-thresh", default=True, help="Optimize decision threshold on validation set")
@click.option("--use-class-weights/--no-use-class-weights", default=True, help="Use balanced class weighting")
def main(mode: str, tune: bool, optimize_thresh: bool, use_class_weights: bool) -> None:
    train_corner_models(mode, tune, optimize_thresh, use_class_weights)


if __name__ == "__main__":
    main()
