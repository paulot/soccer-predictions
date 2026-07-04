import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from click.testing import CliRunner
from download_data import calculate_time_weight, build_self_contained_pipeline, main


class TestDownloadData(unittest.TestCase):
    def test_calculate_time_weight(self):
        """Test exponential time decay weight calculation."""
        ref_date = pd.to_datetime("2026-06-30")
        decay_lambda = 0.0019

        # Same date should give weight 1.0
        weight_same = calculate_time_weight("2026-06-30", ref_date, decay_lambda)
        self.assertAlmostEqual(weight_same, 1.0)

        # Past date should give weight < 1.0
        weight_past = calculate_time_weight("2025-06-30", ref_date, decay_lambda)
        self.assertLess(weight_past, 1.0)
        self.assertGreater(weight_past, 0.0)

        # Future date should be capped at 0 days ago (weight 1.0)
        weight_future = calculate_time_weight("2027-06-30", ref_date, decay_lambda)
        self.assertAlmostEqual(weight_future, 1.0)

        # Invalid date string should fallback to 1.0
        weight_invalid = calculate_time_weight("invalid-date", ref_date, decay_lambda)
        self.assertEqual(weight_invalid, 1.0)

    @patch("download_data.sb.competitions")
    @patch("download_data.sb.matches")
    @patch("download_data.sb.events")
    @patch("download_data.pd.DataFrame.to_csv")
    @patch("download_data.os.makedirs")
    @patch("download_data.os.path.exists")
    @patch("builtins.print")
    def test_build_self_contained_pipeline(
        self, mock_print, mock_exists, mock_makedirs, mock_to_csv, mock_events, mock_matches, mock_competitions
    ):
        """Test the full data download and profile compilation pipeline."""
        mock_exists.return_value = False  # Force downloading events rather than reading cache

        mock_competitions.return_value = pd.DataFrame(
            {
                "competition_name": ["FIFA World Cup"],
                "competition_id": [43],
                "season_name": ["2022"],
                "season_id": [106],
            }
        )

        mock_matches.return_value = pd.DataFrame(
            {
                "match_id": [3869685],
                "match_date": ["2022-12-18"],
                "home_team": ["Argentina"],
                "away_team": ["France"],
                "home_managers": ["Lionel Scaloni"],
                "away_managers": ["Didier Deschamps"],
            }
        )

        mock_events.return_value = pd.DataFrame(
            {
                "type": ["Pass", "Shot", "Goal Keeper", "Duel", "Pass"],
                "location": [[50.0, 40.0], [100.0, 40.0], [5.0, 40.0], [60.0, 30.0], [55.0, 45.0]],
                "pass_end_location": [[60.0, 40.0], np.nan, np.nan, np.nan, [65.0, 45.0]],
                "player": ["Messi", "Messi", "Lloris", "De Paul", "Messi"],
                "team": ["Argentina", "Argentina", "France", "Argentina", "Argentina"],
                "pass_outcome": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "shot_outcome": [np.nan, "Goal", np.nan, np.nan, np.nan],
                "goalkeeper_type": [np.nan, np.nan, "Shot Faced", np.nan, np.nan],
                "goalkeeper_outcome": [np.nan, np.nan, "Goal Conceded", np.nan, np.nan],
                "possession": [1, 1, 1, 2, 2],
            }
        )

        build_self_contained_pipeline(max_matches_per_comp=1)

        mock_competitions.assert_called_once()
        mock_matches.assert_called_once_with(competition_id=43, season_id=106)
        self.assertTrue(mock_to_csv.called)

    @patch("download_data.sb.competitions")
    @patch("builtins.print")
    def test_build_self_contained_pipeline_empty(self, mock_print, mock_competitions):
        """Test pipeline behavior when no matching competitions/passes are found."""
        mock_competitions.return_value = pd.DataFrame(
            {
                "competition_name": ["Unknown League"],
                "competition_id": [99],
                "season_name": ["2022"],
                "season_id": [100],
            }
        )
        build_self_contained_pipeline(max_matches_per_comp=1)
        mock_print.assert_any_call("No pass data was successfully downloaded.")

    @patch("download_data.build_self_contained_pipeline")
    def test_main_cli(self, mock_pipeline):
        """Test download_data main CLI command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--max-matches", "5", "--ref-date", "2026-01-01", "--decay-lambda", "0.002"])
        self.assertEqual(result.exit_code, 0)
        mock_pipeline.assert_called_once_with(max_matches_per_comp=5, ref_date="2026-01-01", decay_lambda=0.002)


if __name__ == "__main__":
    unittest.main()
