import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from ml_model.corners.features import extract_corner_features


class TestCornerFeatures(unittest.TestCase):
    @patch("ml_model.corners.features.pd.DataFrame.to_csv")
    @patch("ml_model.corners.features.os.path.exists")
    @patch("ml_model.corners.features.os.listdir")
    @patch("ml_model.corners.features.pd.read_csv")
    def test_extract_corner_features(self, mock_read_csv, mock_listdir, mock_exists, mock_to_csv):
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["match_1.csv"]

        # Create mock DataFrame for a match with a corner kick
        mock_df = pd.DataFrame(
            {
                "type": ["Pass", "Shot"],
                "play_pattern": ["From Corner", "From Corner"],
                "pass_type": ["Corner", None],
                "team": ["Spain", "Spain"],
                "player": ["Daniel Olmo", "Pedri"],
                "location": ["[120.0, 0.1]", "[111.0, 35.0]"],
                "pass_end_location": ["[111.0, 35.0]", None],
                "pass_outcome": [None, None],
                "minute": [8.0, 8.0],
                "under_pressure": [False, True],
                "shot_outcome": [None, "Goal"],
            }
        )
        mock_read_csv.return_value = mock_df

        df_out = extract_corner_features(mode="iteration")
        self.assertIsInstance(df_out, pd.DataFrame)
        self.assertFalse(df_out.empty)
        self.assertIn("target_routine", df_out.columns)
        self.assertIn("target_outcome", df_out.columns)
        self.assertIn("routine_lag_1", df_out.columns)
        self.assertIn("hist_rate_routine_3", df_out.columns)
        self.assertIn("hist_rate_routine_1", df_out.columns)
        self.assertIn("consecutive_same_routine", df_out.columns)
        self.assertIn("delivery_distance", df_out.columns)
        self.assertIn("end_zone", df_out.columns)


if __name__ == "__main__":
    unittest.main()
