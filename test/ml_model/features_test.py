import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from click.testing import CliRunner
from ml_model.features import parse_timestamp_to_seconds, extract_features_and_targets, main


class TestMLFeatures(unittest.TestCase):
    def test_parse_timestamp_to_seconds(self):
        """Test timestamp parsing to total seconds."""
        self.assertAlmostEqual(parse_timestamp_to_seconds("00:01:30.500"), 90.5)
        self.assertAlmostEqual(parse_timestamp_to_seconds("01:10:00.000"), 4200.0)
        self.assertEqual(parse_timestamp_to_seconds(None), 0.0)
        self.assertEqual(parse_timestamp_to_seconds(np.nan), 0.0)
        self.assertEqual(parse_timestamp_to_seconds("invalid-time"), 0.0)

    @patch("ml_model.features.pd.read_csv")
    @patch("ml_model.features.os.path.exists")
    @patch("ml_model.features.os.listdir")
    @patch("ml_model.features.pickle.load")
    @patch("ml_model.features.pd.DataFrame.to_csv")
    @patch("ml_model.features.json.dump")
    @patch("ml_model.features.os.makedirs")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_extract_features_and_targets(
        self,
        mock_print,
        mock_open,
        mock_makedirs,
        mock_dump,
        mock_to_csv,
        mock_load,
        mock_listdir,
        mock_exists,
        mock_read_csv,
    ):
        """Test feature extraction and target construction from raw event files."""
        mock_exists.return_value = True
        mock_listdir.return_value = ["1001.csv"]

        # Mock embeddings
        mock_load.side_effect = [
            {"Z_2_2": np.zeros(4), "Z_3_2": np.zeros(4)},  # zone_embeddings
            {"Player A": np.zeros(8), "Player B": np.zeros(8)},  # player_embeddings
            {"Manager Home": np.zeros(4), "Manager Away": np.zeros(4)},  # manager_embeddings
        ]

        def side_effect_read_csv(filepath, *args, **kwargs):
            if "statsbomb_player_profiles" in str(filepath):
                return pd.DataFrame([{"player": "Player A", "accuracy": 0.85, "progressive_ratio": 0.4}])
            elif "goalkeeper_profiles" in str(filepath):
                return pd.DataFrame([{"goalkeeper": "GK Home", "save_ratio": 0.75}])
            elif "team_defensive_profiles" in str(filepath):
                return pd.DataFrame([{"team": "Home Team", "zone": "Z_2_2", "defensive_rate": 0.1}])
            elif "manager_profiles" in str(filepath):
                return pd.DataFrame([{"manager": "Manager Home", "directness": 5.0, "width": 5.0}])
            elif "1001.csv" in str(filepath):
                return pd.DataFrame(
                    {
                        "period": [1, 1],
                        "timestamp": ["00:05:00.0", "00:05:03.0"],
                        "team": ["Home Team", "Away Team"],
                        "player": ["Player A", "Player A"],
                        "type": ["Pass", "Pass"],
                        "location": ["[50.0, 40.0]", "[60.0, 40.0]"],
                        "pass_end_location": ["[60.0, 40.0]", "[70.0, 40.0]"],
                        "possession": [1, 1],
                        "position": ["Center Midfield", "Center Midfield"],
                        "pass_outcome": [np.nan, "Incomplete"],
                        "under_pressure": [True, False],
                    }
                )
            return pd.DataFrame()

        mock_read_csv.side_effect = side_effect_read_csv

        extract_features_and_targets("iteration")
        self.assertTrue(mock_to_csv.called)
        self.assertTrue(mock_dump.called)

    @patch("ml_model.features.pd.read_csv")
    @patch("builtins.print")
    def test_extract_features_and_targets_error(self, mock_print, mock_read_csv):
        """Test feature extraction when loading profile datasets fails."""
        mock_read_csv.side_effect = Exception("File missing")
        extract_features_and_targets("iteration")
        mock_print.assert_any_call("Error loading profiles: File missing. Please run download_data.py first.")

    @patch("ml_model.features.extract_features_and_targets")
    def test_main_cli(self, mock_extract):
        """Test ml_model features main CLI command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--mode", "iteration"])
        self.assertEqual(result.exit_code, 0)
        mock_extract.assert_called_once_with("iteration")


if __name__ == "__main__":
    unittest.main()
