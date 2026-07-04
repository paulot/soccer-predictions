import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from click.testing import CliRunner
from ml_model.backtest import run_single_simulation, run_ml_backtest, main
from mcmc_simulation import build_30_zone_grid


class TestMLBacktest(unittest.TestCase):
    def setUp(self):
        self.zones = build_30_zone_grid()
        self.base_matrix = pd.DataFrame(1.0 / 30.0, index=self.zones, columns=self.zones)
        self.df_events = pd.DataFrame(
            {
                "start_zone": ["Z_2_2", "Z_3_2"],
                "player": ["Player A", "Player B"],
                "team": ["Home Team", "Away Team"],
                "type": ["Pass", "Pass"],
                "location": ["[50.0, 40.0]", "[60.0, 40.0]"],
                "pass_end_location": ["[60.0, 40.0]", "[70.0, 40.0]"],
            }
        )
        self.player_profiles = {"Player A": {"accuracy": 0.85, "progressive_ratio": 0.4}}
        self.gk_profiles = {"GK Home": 0.75}
        self.team_defensive_profiles = {"Home Team": {"Z_2_2": 0.1}}
        self.manager_profiles = {"Manager Home": {"directness": 5.0, "width": 5.0, "tempo": 5.0}}
        self.team_to_manager = {"Home Team": "Manager Home"}
        self.player_to_team = {"Player A": "Home Team"}

    @patch("ml_model.backtest.simulate_full_match", return_value=(2, 1))
    def test_run_single_simulation_heuristic(self, mock_sim):
        """Test running a single simulation worker using heuristic model."""
        args = (
            "heuristic",
            "",
            "",
            self.base_matrix,
            "Home Team",
            "Away Team",
            self.df_events,
            self.player_profiles,
            self.gk_profiles,
            self.team_defensive_profiles,
            self.manager_profiles,
            self.team_to_manager,
            self.player_to_team,
            self.zones,
            10,
        )
        res = run_single_simulation(args)
        self.assertEqual(res, (2, 1))
        mock_sim.assert_called_once()

    @patch("ml_model.backtest.sb.competitions")
    @patch("ml_model.backtest.sb.matches")
    @patch("ml_model.backtest.sb.events")
    @patch("ml_model.backtest.pd.read_csv")
    @patch("ml_model.backtest.os.path.exists")
    @patch("ml_model.backtest.ProcessPoolExecutor")
    @patch("builtins.print")
    def test_run_ml_backtest_heuristic(
        self, mock_print, mock_pool, mock_exists, mock_read_csv, mock_events, mock_matches, mock_competitions
    ):
        """Test running ML backtest pipeline with heuristic model."""
        mock_exists.return_value = True

        def side_effect_read_csv(filepath, *args, **kwargs):
            if "global_baseline_matrix" in str(filepath):
                return self.base_matrix.copy()
            elif "statsbomb_player_profiles" in str(filepath):
                return pd.DataFrame([{"player": "Player A", "accuracy": 0.85, "progressive_ratio": 0.4}])
            elif "goalkeeper_profiles" in str(filepath):
                return pd.DataFrame([{"goalkeeper": "GK Home", "save_ratio": 0.75}])
            elif "team_defensive_profiles" in str(filepath):
                return pd.DataFrame([{"team": "Home Team", "zone": "Z_2_2", "defensive_rate": 0.1}])
            elif "manager_profiles" in str(filepath):
                return pd.DataFrame([{"manager": "Manager Home", "directness": 5.0, "width": 5.0, "tempo": 5.0}])
            elif "raw_events" in str(filepath):
                return pd.DataFrame(
                    {
                        "team": ["Home Team", "Away Team"],
                        "player": ["Player A", "Player B"],
                        "type": ["Pass", "Shot"],
                        "location": ["[50.0, 40.0]", "[100.0, 40.0]"],
                        "pass_end_location": ["[60.0, 40.0]", np.nan],
                        "shot_outcome": [np.nan, "Goal"],
                        "possession": [1, 1],
                    }
                )
            return pd.DataFrame()

        mock_read_csv.side_effect = side_effect_read_csv

        mock_competitions.return_value = pd.DataFrame(
            {
                "competition_name": ["FIFA World Cup"],
                "season_name": ["2022"],
                "competition_id": [43],
                "season_id": [106],
            }
        )
        mock_matches.return_value = pd.DataFrame({"match_id": [1001]})

        # Mock ProcessPoolExecutor map
        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.map.return_value = [(2, 1), (1, 1)]  # 1 win, 1 draw
        mock_pool.return_value = mock_executor_instance

        res = run_ml_backtest("heuristic", "iteration", num_simulations=2)
        self.assertIsInstance(res, dict)
        self.assertIn("brier", res)
        self.assertIn("log_loss", res)
        self.assertIn("accuracy", res)

    @patch("ml_model.backtest.pd.read_csv")
    @patch("builtins.print")
    def test_run_ml_backtest_load_error(self, mock_print, mock_read_csv):
        """Test ML backtest when global dataset loading fails."""
        mock_read_csv.side_effect = Exception("Missing dataset")
        res = run_ml_backtest("heuristic", "iteration")
        self.assertIsNone(res)
        mock_print.assert_any_call("Error loading global datasets: Missing dataset. Please run download_data.py first.")

    @patch("ml_model.backtest.run_ml_backtest")
    def test_main_cli(self, mock_backtest):
        """Test ml_model backtest CLI command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--model", "random_forest", "--mode", "iteration", "--sims", "10"])
        self.assertEqual(result.exit_code, 0)
        mock_backtest.assert_called_once_with("random_forest", "iteration", 10)


if __name__ == "__main__":
    unittest.main()
