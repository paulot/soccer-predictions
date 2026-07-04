import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from click.testing import CliRunner
from ml_model.train_embeddings import train_all_embeddings, main


class TestMLTrainEmbeddings(unittest.TestCase):
    @patch("ml_model.train_embeddings.os.listdir")
    @patch("ml_model.train_embeddings.pd.read_csv")
    @patch("ml_model.train_embeddings.os.makedirs")
    @patch("ml_model.train_embeddings.pickle.dump")
    @patch("builtins.open")
    @patch("builtins.print")
    def test_train_all_embeddings(self, mock_print, mock_open, mock_dump, mock_makedirs, mock_read_csv, mock_listdir):
        """Test computing and saving spectral embeddings for zones, players, and managers."""
        mock_listdir.return_value = ["match_1001.csv", "match_1002.csv"]

        mock_df_1 = pd.DataFrame(
            {
                "type": ["Pass", "Pass", "Pass"],
                "location": ["[10.0, 20.0]", "[30.0, 40.0]", "[50.0, 60.0]"],
                "pass_end_location": ["[30.0, 40.0]", "[50.0, 60.0]", "[70.0, 80.0]"],
                "pass_outcome": [np.nan, "Incomplete", np.nan],
                "player": ["Player A", "Player A", "Player B"],
                "team": ["Canada", "Canada", "Morocco"],
            }
        )
        mock_df_2 = pd.DataFrame(
            {
                "type": ["Pass"],
                "location": ["[20.0, 30.0]"],
                "pass_end_location": ["[40.0, 50.0]"],
                "pass_outcome": [np.nan],
                "player": ["Player B"],
                "team": ["Morocco"],
            }
        )
        mock_read_csv.side_effect = [mock_df_1, mock_df_2]

        train_all_embeddings("iteration")

        self.assertEqual(mock_dump.call_count, 3)
        mock_makedirs.assert_called_with("data/embeddings", exist_ok=True)

    @patch("ml_model.train_embeddings.TruncatedSVD")
    @patch("ml_model.train_embeddings.os.listdir")
    @patch("builtins.print")
    def test_train_all_embeddings_no_files(self, mock_print, mock_listdir, mock_svd):
        """Test embedding training when raw events directory is empty."""
        mock_listdir.return_value = []
        mock_svd_zone = MagicMock()
        mock_svd_zone.fit_transform.return_value = np.zeros((30, 4))
        mock_svd_other = MagicMock()
        mock_svd_other.fit_transform.return_value = np.zeros((0, 4))
        mock_svd.side_effect = [mock_svd_zone, mock_svd_other, mock_svd_other]

        with patch("ml_model.train_embeddings.pickle.dump") as mock_dump, patch("builtins.open"):
            train_all_embeddings("iteration")
            self.assertEqual(mock_dump.call_count, 3)

    @patch("ml_model.train_embeddings.train_all_embeddings")
    def test_main_cli(self, mock_train_embs):
        """Test ml_model train_embeddings main CLI command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--mode", "iteration"])
        self.assertEqual(result.exit_code, 0)
        mock_train_embs.assert_called_once_with("iteration")


if __name__ == "__main__":
    unittest.main()
