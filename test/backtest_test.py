import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from backtest import simulate_full_match, run_loocv_backtest
from mcmc_simulation import build_30_zone_grid


class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.zones = build_30_zone_grid()
        self.base_matrix = pd.DataFrame(1.0 / 30.0, index=self.zones, columns=self.zones)
        self.df_events = pd.DataFrame(
            {
                "start_zone": ["Z_2_2", "Z_3_2", "Z_4_2"],
                "player": ["Player A", "Player B", "Player C"],
                "team": ["Home Team", "Home Team", "Away Team"],
                "type": ["Pass", "Pass", "Pass"],
            }
        )
        self.player_profiles = {
            "Player A": {"accuracy": 0.85, "progressive_ratio": 0.4, "shot_conversion": 0.15},
            "Player B": {"accuracy": 0.80, "progressive_ratio": 0.3, "shot_conversion": 0.10},
            "Player C": {"accuracy": 0.75, "progressive_ratio": 0.2, "shot_conversion": 0.20},
        }
        self.gk_profiles = {"GK Home": 0.75, "GK Away": 0.70}
        self.team_defensive_profiles = {
            "Home Team": {"Z_2_2": 0.1, "Z_3_2": 0.15},
            "Away Team": {"Z_2_2": 0.12, "Z_3_2": 0.18},
        }
        self.manager_profiles = {
            "Manager Home": {"directness": 6.0, "width": 5.0, "tempo": 5.0},
            "Manager Away": {"directness": 4.0, "width": 6.0, "tempo": 7.0},
        }
        self.team_to_manager = {"Home Team": "Manager Home", "Away Team": "Manager Away"}
        self.player_to_team = {
            "Player A": "Home Team",
            "Player B": "Home Team",
            "GK Home": "Home Team",
            "Player C": "Away Team",
            "GK Away": "Away Team",
        }

    def test_simulate_full_match(self):
        """Test simulating a full football match between two teams."""
        home_goals, away_goals = simulate_full_match(
            home_team="Home Team",
            away_team="Away Team",
            base_matrix=self.base_matrix,
            df_events=self.df_events,
            player_profiles=self.player_profiles,
            gk_profiles=self.gk_profiles,
            team_defensive_profiles=self.team_defensive_profiles,
            manager_profiles=self.manager_profiles,
            team_to_manager=self.team_to_manager,
            player_to_team=self.player_to_team,
            zones=self.zones,
            num_possessions=10,
        )
        self.assertIsInstance(home_goals, int)
        self.assertIsInstance(away_goals, int)
        self.assertGreaterEqual(home_goals, 0)
        self.assertGreaterEqual(away_goals, 0)

    @patch("backtest.sb.events")
    @patch("backtest.pd.read_csv")
    @patch("os.path.exists")
    @patch("builtins.print")
    def test_run_loocv_backtest(self, mock_print, mock_exists, mock_read_csv, mock_events):
        """Test running LOOCV backtest pipeline over a set of match IDs."""
        mock_exists.return_value = True

        # Mock reading CSVs for global datasets and cached events
        def side_effect_read_csv(filepath, *args, **kwargs):
            if "global_baseline_matrix" in str(filepath):
                return self.base_matrix.copy()
            elif "statsbomb_player_profiles" in str(filepath):
                return pd.DataFrame(
                    [{"player": "Player A", "accuracy": 0.85, "progressive_ratio": 0.4, "shot_conversion": 0.15}]
                )
            elif "goalkeeper_profiles" in str(filepath):
                return pd.DataFrame([{"goalkeeper": "GK Home", "save_ratio": 0.75}])
            elif "team_defensive_profiles" in str(filepath):
                return pd.DataFrame([{"team": "Home Team", "zone": "Z_2_2", "defensive_rate": 0.1}])
            elif "manager_profiles" in str(filepath):
                return pd.DataFrame([{"manager": "Manager Home", "directness": 5.0, "width": 5.0, "tempo": 5.0}])
            elif "raw_events" in str(filepath):
                return pd.DataFrame(
                    {
                        "team": ["Home Team", "Away Team", "Home Team"],
                        "player": ["Player A", "Player C", "Player A"],
                        "type": ["Pass", "Pass", "Shot"],
                        "location": ["[50.0, 40.0]", "[60.0, 40.0]", "[100.0, 40.0]"],
                        "pass_end_location": ["[60.0, 40.0]", "[70.0, 40.0]", np.nan],
                        "shot_outcome": [np.nan, np.nan, "Goal"],
                    }
                )
            return pd.DataFrame()

        mock_read_csv.side_effect = side_effect_read_csv

        # Run backtest with 2 simulations on match ID 1001
        run_loocv_backtest([1001], num_simulations=2)
        mock_read_csv.assert_called()

    @patch("backtest.pd.read_csv")
    @patch("builtins.print")
    def test_run_loocv_backtest_load_error(self, mock_print, mock_read_csv):
        """Test LOOCV backtest graceful exit when global dataset loading fails."""
        mock_read_csv.side_effect = Exception("File not found")
        run_loocv_backtest([1001], num_simulations=2)
        mock_print.assert_any_call("Error loading global datasets: File not found")


if __name__ == "__main__":
    unittest.main()
