import unittest
from unittest.mock import patch
from click.testing import CliRunner
from run_pipeline import cli


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test the main CLI group help message."""
        result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Unified CLI for MCMC Soccer Prediction Pipeline", result.output)

    @patch("run_pipeline.train_models")
    def test_train_command(self, mock_train_models):
        """Test the train CLI command."""
        result = self.runner.invoke(cli, ["train", "--model", "random_forest", "--mode", "iteration"])
        self.assertEqual(result.exit_code, 0)
        mock_train_models.assert_called_once_with("random_forest", "iteration")
        self.assertIn("DEBUG: train_models finished", result.output)

    @patch("run_pipeline.run_ml_backtest")
    def test_backtest_command(self, mock_run_ml_backtest):
        """Test the backtest CLI command."""
        result = self.runner.invoke(cli, ["backtest", "--model", "xgboost", "--mode", "production", "--sims", "100"])
        self.assertEqual(result.exit_code, 0)
        mock_run_ml_backtest.assert_called_once_with("xgboost", "production", 100)


if __name__ == "__main__":
    unittest.main()
