import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import torch
from click.testing import CliRunner
from ml_model.train import train_models, main


class TestMLTrain(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Construct dummy training dataset matching required features
        dest_features = [
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

        outcome_features = dest_features + ["pass_length", "pass_angle", "pressure_differential"]
        all_cols = outcome_features + ["outcome", "end_zone_x", "end_zone_y"]

        # 40 rows of random data
        data = np.random.randn(40, len(all_cols))
        self.df_train = pd.DataFrame(data, columns=all_cols)
        self.df_train["outcome"] = [0, 1] * 20
        self.df_train["end_zone_x"] = np.random.choice(range(6), size=40)
        self.df_train["end_zone_y"] = np.random.choice(range(5), size=40)
        self.df_train["player_role"] = np.random.choice([0, 1, 2, 3], size=40)
        self.df_train["start_zone_x"] = np.random.choice(range(6), size=40)
        self.df_train["start_zone_y"] = np.random.choice(range(5), size=40)

    @patch("ml_model.train.log_loss", return_value=0.5)
    @patch("ml_model.train.accuracy_score", return_value=0.8)
    @patch("ml_model.train.pd.read_csv")
    @patch("ml_model.train.os.path.exists")
    @patch("ml_model.train.os.makedirs")
    @patch("ml_model.train.pickle.dump")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_train_models_random_forest(
        self, mock_print, mock_open, mock_dump, mock_makedirs, mock_exists, mock_read_csv, mock_acc, mock_loss
    ):
        """Test training random forest classifier models."""
        mock_exists.return_value = True
        mock_read_csv.return_value = self.df_train

        train_models("random_forest", "iteration")
        self.assertTrue(mock_dump.called)

    @patch("ml_model.train.log_loss", return_value=0.5)
    @patch("ml_model.train.accuracy_score", return_value=0.8)
    @patch("ml_model.train.pd.read_csv")
    @patch("ml_model.train.os.path.exists")
    @patch("ml_model.train.os.makedirs")
    @patch("ml_model.train.pickle.dump")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_train_models_logistic_regression(
        self, mock_print, mock_open, mock_dump, mock_makedirs, mock_exists, mock_read_csv, mock_acc, mock_loss
    ):
        """Test training logistic regression classifier models."""
        mock_exists.return_value = True
        mock_read_csv.return_value = self.df_train

        train_models("logistic_regression", "iteration")
        self.assertTrue(mock_dump.called)

    @patch("torch.jit.script", side_effect=lambda x: x)
    @patch("torch.jit.save")
    @patch("ml_model.train.log_loss", return_value=0.5)
    @patch("ml_model.train.accuracy_score", return_value=0.8)
    @patch("ml_model.train.pd.read_csv")
    @patch("ml_model.train.os.path.exists")
    @patch("ml_model.train.os.makedirs")
    @patch("ml_model.train.pickle.dump")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_train_models_neural_network(
        self,
        mock_print,
        mock_open,
        mock_dump,
        mock_makedirs,
        mock_exists,
        mock_read_csv,
        mock_acc,
        mock_loss,
        mock_jit_save,
        mock_jit_script,
    ):
        """Test training neural network classifier models with PyTorch."""
        mock_exists.return_value = True
        mock_read_csv.return_value = self.df_train

        with (
            patch("ml_model.pytorch_models.OutcomeNN") as mock_outcome_class,
            patch("ml_model.pytorch_models.DestinationNN") as mock_dest_class,
        ):
            mock_outcome_inst = MagicMock()
            mock_outcome_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
            mock_outcome_inst.return_value = torch.tensor([[0.5]] * 32, requires_grad=True)
            mock_outcome_class.return_value = mock_outcome_inst

            mock_dest_inst = MagicMock()
            mock_dest_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
            mock_dest_inst.return_value = torch.randn(16, 30, requires_grad=True)
            mock_dest_class.return_value = mock_dest_inst

            train_models("neural_network", "iteration")
            self.assertTrue(mock_jit_save.called)
            self.assertTrue(mock_dump.called)

    @patch("ml_model.train.log_loss", return_value=0.5)
    @patch("ml_model.train.accuracy_score", return_value=0.8)
    @patch("ml_model.features.extract_features_and_targets")
    @patch("ml_model.train.pd.read_csv")
    @patch("ml_model.train.os.path.exists")
    @patch("ml_model.train.os.makedirs")
    @patch("ml_model.train.pickle.dump")
    @patch("builtins.open")
    def test_train_models_missing_data(
        self, mock_open, mock_dump, mock_makedirs, mock_exists, mock_read_csv, mock_extract, mock_acc, mock_loss
    ):
        """Test training triggers feature extraction when training CSV is missing."""
        mock_exists.side_effect = [False, True]
        mock_read_csv.return_value = self.df_train

        train_models("random_forest", "iteration")
        mock_extract.assert_called_once_with("iteration")

    @patch("ml_model.train.train_models")
    def test_main_cli(self, mock_train):
        """Test ml_model train main CLI command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--model", "random_forest", "--mode", "iteration"])
        self.assertEqual(result.exit_code, 0)
        mock_train.assert_called_once_with("random_forest", "iteration")


if __name__ == "__main__":
    unittest.main()
