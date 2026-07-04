import unittest
from unittest.mock import MagicMock
import pandas as pd
from ml_model.simulator import simulate_full_match
from mcmc_simulation import build_30_zone_grid


class TestMLSimulator(unittest.TestCase):
    def setUp(self):
        self.zones = build_30_zone_grid()
        self.df_events = pd.DataFrame(
            {
                "start_zone": ["Z_2_2", "Z_3_2", "Z_5_2"],
                "player": ["Player A", "Player B", "Player C"],
                "team": ["Home Team", "Home Team", "Away Team"],
            }
        )
        self.player_profiles = {
            "Player A": {"accuracy": 0.85, "progressive_ratio": 0.4, "shot_conversion": 0.20},
            "Player B": {"accuracy": 0.80, "progressive_ratio": 0.3, "shot_conversion": 0.15},
            "Player C": {"accuracy": 0.75, "progressive_ratio": 0.2, "shot_conversion": 0.25},
        }
        self.gk_profiles = {"GK Home": 0.75, "GK Away": 0.70}
        self.team_defensive_profiles = {"Home Team": {"Z_2_2": 0.1}, "Away Team": {"Z_2_2": 0.15}}
        self.manager_profiles = {"Manager Home": {"directness": 6.0}, "Manager Away": {"directness": 4.0}}
        self.team_to_manager = {"Home Team": "Manager Home", "Away Team": "Manager Away"}
        self.player_to_team = {
            "Player A": "Home Team",
            "Player B": "Home Team",
            "GK Home": "Home Team",
            "Player C": "Away Team",
            "GK Away": "Away Team",
        }

    def test_simulate_full_match(self):
        """Test full match simulation using modular transition models."""
        mock_model = MagicMock()
        # Return equal transition probability for all zones
        mock_model.get_transition_probabilities.return_value = pd.Series(1.0 / 30.0, index=self.zones)
        # Return 20% turnover probability
        mock_model.get_turnover_probability.return_value = 0.20

        home_goals, away_goals = simulate_full_match(
            home_team="Home Team",
            away_team="Away Team",
            transition_model=mock_model,
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
        mock_model.get_transition_probabilities.assert_called()

    def test_simulate_full_match_loop_prevention(self):
        """Test that infinite possession loop is prevented when turnover prob is 0."""
        mock_model = MagicMock()
        mock_model.get_transition_probabilities.return_value = pd.Series(1.0 / 30.0, index=self.zones)
        # 0% turnover probability would cause infinite loop without safety check
        mock_model.get_turnover_probability.return_value = 0.0

        home_goals, away_goals = simulate_full_match(
            home_team="Home Team",
            away_team="Away Team",
            transition_model=mock_model,
            df_events=self.df_events,
            player_profiles=self.player_profiles,
            gk_profiles=self.gk_profiles,
            team_defensive_profiles=self.team_defensive_profiles,
            manager_profiles=self.manager_profiles,
            team_to_manager=self.team_to_manager,
            player_to_team=self.player_to_team,
            zones=self.zones,
            num_possessions=2,
        )
        # Should terminate cleanly due to max sequence index check (>= 50)
        self.assertIsInstance(home_goals, int)
        self.assertIsInstance(away_goals, int)


if __name__ == "__main__":
    unittest.main()
