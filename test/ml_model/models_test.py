import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from ml_model.models import BaseTransitionModel, HeuristicTransitionModel, MLTransitionModel
from mcmc_simulation import build_30_zone_grid


class TestMLModels(unittest.TestCase):
    def setUp(self):
        self.zones = build_30_zone_grid()
        self.base_matrix = pd.DataFrame(1.0 / 30.0, index=self.zones, columns=self.zones)
        self.player_profiles = {"Player A": {"accuracy": 0.85, "progressive_ratio": 0.5}}
        self.manager_profiles = {"Manager Home": {"directness": 7.0, "width": 3.0, "tempo": 6.0}}
        self.team_to_manager = {"Home Team": "Manager Home"}
        self.player_to_team = {"Player A": "Home Team"}
        self.team_defensive_profiles = {"Away Team": {"Z_2_2": 0.15}}

    def test_base_transition_model(self):
        """Test that BaseTransitionModel raises NotImplementedError for abstract methods."""
        model = BaseTransitionModel()
        with self.assertRaises(NotImplementedError):
            model.get_transition_probabilities("Z_2_2", None, {}, {}, {}, {}, "Home Team", "Away Team", self.zones)
        with self.assertRaises(NotImplementedError):
            model.get_turnover_probability("Z_2_2", None, {}, {}, {}, "Home Team", "Away Team")

    def test_heuristic_transition_model(self):
        """Test HeuristicTransitionModel probability calculations and modifiers."""
        model = HeuristicTransitionModel(self.base_matrix)

        # Test transition probabilities
        probs = model.get_transition_probabilities(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.manager_profiles,
            self.team_to_manager,
            self.player_to_team,
            "Home Team",
            "Away Team",
            self.zones,
        )
        self.assertEqual(len(probs), 30)
        self.assertAlmostEqual(probs.sum(), 1.0)

        # Test turnover probability
        turnover = model.get_turnover_probability(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.team_defensive_profiles,
            self.player_to_team,
            "Home Team",
            "Away Team",
        )
        self.assertGreater(turnover, 0.0)
        self.assertLess(turnover, 1.0)

    @patch("ml_model.models.pickle.load")
    @patch("ml_model.models.pd.read_csv")
    @patch("ml_model.models.json.load")
    @patch("builtins.open")
    def test_ml_transition_model(self, mock_open, mock_json_load, mock_read_csv, mock_pickle_load):
        """Test MLTransitionModel initialization, feature compilation, and caching."""
        # Mock scikit-learn models
        mock_outcome_model = MagicMock(spec=["predict_proba", "classes_"])
        mock_outcome_model.predict_proba.return_value = np.array([[0.8, 0.2]])

        mock_dest_model = MagicMock(spec=["predict_proba", "classes_"])
        mock_dest_model.predict_proba.return_value = np.array([[1.0 / 30.0] * 30])
        mock_dest_model.classes_ = list(range(30))

        # Setup pickle loads for models and embeddings
        mock_pickle_load.side_effect = [
            mock_outcome_model,
            mock_dest_model,
            {"Z_2_2": np.zeros(4)},  # zone_embeddings
            {"Player A": np.zeros(8)},  # player_embeddings
            {"Manager Home": np.zeros(4)},  # manager_embeddings
        ]

        mock_read_csv.side_effect = [
            pd.DataFrame([{"goalkeeper": "GK Home", "save_ratio": 0.75}]),
            pd.DataFrame([{"team": "Away Team", "zone": "Z_2_2", "defensive_rate": 0.15}]),
        ]
        mock_json_load.return_value = {"Player A": 2}

        model = MLTransitionModel("fake_outcome.pkl", "fake_dest.pkl")

        # Test feature compilation
        feats = model._compile_features(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.manager_profiles,
            self.team_to_manager,
            self.player_to_team,
            "Home Team",
            "Away Team",
        )
        self.assertIn("start_zone_x", feats)
        self.assertIn("opp_defensive_rate", feats)
        self.assertEqual(feats["start_zone_x"], 2.0)

        # Test transition probabilities and caching
        probs1 = model.get_transition_probabilities(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.manager_profiles,
            self.team_to_manager,
            self.player_to_team,
            "Home Team",
            "Away Team",
            self.zones,
        )
        self.assertAlmostEqual(probs1.sum(), 1.0)

        # Second call should use cache
        probs2 = model.get_transition_probabilities(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.manager_profiles,
            self.team_to_manager,
            self.player_to_team,
            "Home Team",
            "Away Team",
            self.zones,
        )
        self.assertIs(probs1, probs2)

        # Test turnover probability and caching
        turnover1 = model.get_turnover_probability(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.team_defensive_profiles,
            self.player_to_team,
            "Home Team",
            "Away Team",
        )
        self.assertAlmostEqual(turnover1, 0.2)

        turnover2 = model.get_turnover_probability(
            "Z_2_2",
            "Player A",
            self.player_profiles,
            self.team_defensive_profiles,
            self.player_to_team,
            "Home Team",
            "Away Team",
        )
        self.assertIs(turnover1, turnover2)


if __name__ == "__main__":
    unittest.main()
