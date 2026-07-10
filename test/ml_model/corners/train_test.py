import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from ml_model.corners.train import train_corner_models


class TestCornerTrain(unittest.TestCase):
    @patch("ml_model.corners.train.os.path.exists")
    @patch("ml_model.corners.train.pd.read_csv")
    @patch("ml_model.corners.train.CornerRoutineXGB")
    @patch("ml_model.corners.train.CornerOutcomeXGB")
    def test_train_corner_models(self, mock_outcome_cls, mock_routine_cls, mock_read_csv, mock_exists):
        mock_exists.return_value = True

        mock_df = pd.DataFrame(
            {
                "is_right_corner": [0, 1, 0, 1] * 10,
                "time_ratio": [0.1, 0.5, 0.8, 0.9] * 10,
                "score_differential": [0, -1, 1, 0] * 10,
                "is_home_team": [1, 0, 1, 0] * 10,
                "inswinging": [1, 1, 0, 0] * 10,
                "taker_accuracy": [0.8] * 40,
                "taker_key_pass_ratio": [0.2] * 40,
                "team_directness": [5.0] * 40,
                "opp_def_rate": [0.15] * 40,
                "under_pressure": [0, 1, 0, 1] * 10,
                "corner_cluster_density": [0, 1, 0, 2] * 10,
                "aerial_height_advantage": [0.05, -0.05, 0.1, 0.0] * 10,
                "goalkeeper_line_command": [0.7] * 40,
                "taker_corner_assist_rate": [0.15] * 40,
                "routine_lag_1": [-1, 0, 1, 2] * 10,
                "routine_lag_2": [-1, 0, 1, 2] * 10,
                "routine_lag_3": [-1, 0, 1, 2] * 10,
                "routine_lag_4": [-1, 0, 1, 2] * 10,
                "routine_lag_5": [-1, 0, 1, 2] * 10,
                "hist_rate_routine_1": [0.15] * 40,
                "hist_rate_routine_2": [0.10] * 40,
                "hist_rate_routine_3": [0.00] * 40,
                "team_match_corner_count": [1, 2, 3, 4] * 10,
                "consecutive_same_routine": [0, 1, 2, 0] * 10,
                "end_zone": ["Z_5_2", "Z_5_1", "Z_5_0", "Z_5_2"] * 10,
                "end_x": [108.0, 114.0, 110.0, 105.0] * 10,
                "end_y": [40.0, 25.0, 10.0, 42.0] * 10,
                "target_routine": [0, 1, 2, 3] * 10,
                "target_outcome": [0, 1, 0, 1] * 10,
            }
        )
        mock_read_csv.return_value = mock_df

        mock_routine_inst = MagicMock()
        mock_routine_inst.predict.return_value = np.zeros(8)
        mock_routine_inst.predict_proba.return_value = np.ones((8, 4)) / 4.0
        mock_routine_inst.tune_hyperparameters.return_value = {"max_depth": 4}
        mock_routine_cls.return_value = mock_routine_inst

        mock_outcome_inst = MagicMock()
        mock_outcome_inst.predict.return_value = np.zeros(8)
        mock_outcome_inst.predict_proba.return_value = np.ones(8) * 0.5
        mock_outcome_inst.optimize_threshold.return_value = 0.45
        mock_outcome_inst.tune_hyperparameters.return_value = {"max_depth": 4}
        mock_outcome_cls.return_value = mock_outcome_inst

        train_corner_models(mode="iteration", tune=True, optimize_thresh=True, use_class_weights=True)

        mock_routine_inst.tune_hyperparameters.assert_called_once()
        mock_routine_inst.fit.assert_called_once()
        mock_outcome_inst.tune_hyperparameters.assert_called_once()
        mock_outcome_inst.fit.assert_called_once()
        mock_outcome_inst.optimize_threshold.assert_called_once()
        mock_routine_inst.save.assert_called_once()
        mock_outcome_inst.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
