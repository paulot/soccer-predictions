import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def multi_focal_loss_obj(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Any:
    """
    Custom Multi-Class Focal Loss objective for XGBoost (gamma=2.0).
    Down-weights easy-to-classify majority examples (e.g., Central Box crosses)
    to focus tree capacity on hard tactical boundary cases across all 4 corridors.
    """
    gamma = 2.0
    exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    n_samples, _ = y_pred.shape
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0
    p_t = np.sum(p * y_onehot, axis=1, keepdims=True)
    p_t = np.clip(p_t, 1e-6, 1.0 - 1e-6)
    M_i = (1.0 - p_t) ** gamma - gamma * p_t * ((1.0 - p_t) ** (gamma - 1.0)) * np.log(p_t)
    grad = M_i * (p - y_onehot)
    hess = np.maximum(M_i * p * (1.0 - p), 1e-6)
    if sample_weight is not None:
        grad *= sample_weight[:, None]
        hess *= sample_weight[:, None]
    return grad, hess


class CornerRoutineXGB:
    """
    XGBoost 4-class classifier predicting Corner Routine:
    0: Short Corner / Near Side Corridor (closest strip to flag: zy=0 for left, zy=4 for right)
    1: 1st Post / Near Post Corridor (zy=1 for left, zy=3 for right)
    2: Center / Central Goalmouth Corridor (zy=2)
    3: 2nd Post / Far Post Corridor (zy in [3, 4] for left, [1, 0] for right)
    """

    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 150,
        colsample_bytree: float = 0.6,
        colsample_bylevel: float = 0.7,
        min_child_weight: int = 15,
        subsample: float = 1.0,
        reg_alpha: float = 1.0,
        reg_lambda: float = 2.0,
    ):
        self.model = xgb.XGBClassifier(
            objective=multi_focal_loss_obj,
            num_class=4,
            eval_metric="mlogloss",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            min_child_weight=min_child_weight,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
        )
        self.feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_class_weights: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = list(X.columns)
        if use_class_weights and sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        cv: int = 3,
        scoring: str = "f1_macro",
    ) -> Dict[str, Any]:
        if param_grid is None:
            param_grid = {
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [100, 150, 200],
                "colsample_bytree": [0.5, 0.6, 0.7],
                "min_child_weight": [10, 15, 20],
            }
        search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return search.best_params_

    def get_feature_importance(self) -> Dict[str, float]:
        if not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "CornerRoutineXGB":
        with open(filepath, "rb") as f:
            return pickle.load(f)


class CornerOutcomeXGB:
    """
    XGBoost binary classifier predicting Corner Outcome:
    1: Attacking Success (Shot / Goal / Assist / Aerial duel won in box)
    0: Defensive Success (Clearance / Goalkeeper catch / Incomplete pass)
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 150,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
    ):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        self.feature_names: List[str] = []
        self.threshold: float = 0.50

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_class_weights: bool = True,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.feature_names = list(X.columns)
        if use_class_weights and self.model.scale_pos_weight is None:
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            if pos_count > 0:
                self.model.set_params(scale_pos_weight=neg_count / pos_count)
        self.model.fit(X, y, sample_weight=sample_weight)

    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, metric: str = "f1") -> float:
        probs = self.predict_proba(X_val)
        best_thresh = 0.50
        best_score = -1.0
        for thresh in np.linspace(0.05, 0.95, 100):
            preds = (probs >= thresh).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = thresh
        self.threshold = float(best_thresh)
        return self.threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.model.predict_proba(X)
        if probs.shape[1] > 1:
            return probs[:, 1]
        return probs.ravel()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        cv: int = 3,
        scoring: str = "f1",
    ) -> Dict[str, Any]:
        if param_grid is None:
            param_grid = {
                "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [100, 150, 200],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "min_child_weight": [1, 3, 5],
            }
        search = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        return search.best_params_

    def get_feature_importance(self) -> Dict[str, float]:
        if not hasattr(self.model, "feature_importances_"):
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "CornerOutcomeXGB":
        with open(filepath, "rb") as f:
            return pickle.load(f)
