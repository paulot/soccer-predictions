import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from mcmc_simulation import (
    build_30_zone_grid,
    fetch_real_statsbomb_data,
    calculate_player_profiles,
    get_zone_players,
    build_baseline_transition_matrix,
    apply_player_modifier,
    simulate_mcmc_possession_chain,
)


class TestMCMCSimulation(unittest.TestCase):
    def test_build_30_zone_grid(self):
        """Test that the 30-zone pitch grid is correctly constructed."""
        zones = build_30_zone_grid()
        self.assertEqual(len(zones), 30)
        self.assertEqual(zones[0], "Z_0_0")
        self.assertEqual(zones[-1], "Z_5_4")
        self.assertIn("Z_2_2", zones)

    @patch("mcmc_simulation.sb.events")
    def test_fetch_real_statsbomb_data(self, mock_events):
        """Test fetching and processing StatsBomb pass event data."""
        mock_df = pd.DataFrame(
            {
                "type": ["Pass", "Pass", "Shot"],
                "location": [[20.0, 20.0], [40.0, 40.0], [100.0, 40.0]],
                "pass_end_location": [[30.0, 30.0], [50.0, 50.0], np.nan],
                "pass_outcome": [np.nan, "Incomplete", np.nan],
            }
        )
        mock_events.return_value = mock_df

        passes = fetch_real_statsbomb_data()
        self.assertEqual(len(passes), 2)
        self.assertIn("start_zone", passes.columns)
        self.assertIn("end_zone", passes.columns)
        self.assertIn("event_type", passes.columns)
        self.assertEqual(passes.iloc[0]["event_type"], "Pass")
        self.assertEqual(passes.iloc[1]["event_type"], "Turnover")

    def test_calculate_player_profiles(self):
        """Test calculating player passing accuracy and progressive ratios."""
        df_events = pd.DataFrame(
            {
                "player": ["Player A"] * 6 + ["Player B"] * 3,
                "event_type": ["Pass"] * 5 + ["Turnover"] + ["Pass"] * 3,
                "start_zone": ["Z_1_2"] * 6 + ["Z_2_2"] * 3,
                "end_zone": ["Z_2_2"] * 4 + ["Z_1_2"] * 2 + ["Z_3_2"] * 3,
            }
        )
        profiles = calculate_player_profiles(df_events)

        # Player A has 6 passes (>= 5 threshold), 5 successful (accuracy ~0.833)
        self.assertIn("Player A", profiles)
        self.assertAlmostEqual(profiles["Player A"]["accuracy"], 5.0 / 6.0)
        self.assertAlmostEqual(profiles["Player A"]["total_passes"], 6.0)

        # Player B has 3 passes (< 5 threshold), should be skipped
        self.assertNotIn("Player B", profiles)

    def test_get_zone_players(self):
        """Test getting normalized historical pass frequencies by player per zone."""
        df_events = pd.DataFrame(
            {
                "start_zone": ["Z_2_2", "Z_2_2", "Z_2_2", "Z_1_1"],
                "player": ["Player A", "Player A", "Player B", "Player C"],
            }
        )
        zone_players = get_zone_players(df_events, "Z_2_2")
        self.assertEqual(len(zone_players), 2)
        self.assertAlmostEqual(zone_players["Player A"], 2.0 / 3.0)
        self.assertAlmostEqual(zone_players["Player B"], 1.0 / 3.0)

        empty_players = get_zone_players(df_events, "Z_5_4")
        self.assertEqual(empty_players, {})

    def test_build_baseline_transition_matrix(self):
        """Test building row-normalized Markov transition matrix."""
        zones = ["Z_0_0", "Z_0_1", "Z_1_0"]
        df_events = pd.DataFrame(
            {
                "event_type": ["Pass", "Pass", "Pass", "Turnover"],
                "start_zone": ["Z_0_0", "Z_0_0", "Z_0_1", "Z_0_0"],
                "end_zone": ["Z_0_1", "Z_1_0", "Z_0_0", "Z_1_0"],
            }
        )
        matrix = build_baseline_transition_matrix(df_events, zones)
        self.assertEqual(matrix.shape, (3, 3))
        # Z_0_0 has 2 successful passes: 1 to Z_0_1 and 1 to Z_1_0
        self.assertAlmostEqual(matrix.loc["Z_0_0", "Z_0_1"], 0.5)
        self.assertAlmostEqual(matrix.loc["Z_0_0", "Z_1_0"], 0.5)
        self.assertAlmostEqual(matrix.loc["Z_0_0"].sum(), 1.0)
        # Z_1_0 has 0 successful passes, should be all 0.0
        self.assertAlmostEqual(matrix.loc["Z_1_0"].sum(), 0.0)

    def test_apply_player_modifier(self):
        """Test adjusting transition probabilities based on progressive passing ratio."""
        zones = ["Z_1_2", "Z_2_2", "Z_0_2"]
        row_probs = pd.Series([0.2, 0.4, 0.4], index=zones)
        profile = {"progressive_ratio": 0.66}  # 2x multiplier for forward passes

        modified = apply_player_modifier(row_probs, profile, "Z_1_2", zones)
        # Z_2_2 is forward (x=2 > x=1), so its weight increases
        self.assertGreater(modified["Z_2_2"], 0.4)
        self.assertAlmostEqual(modified.sum(), 1.0)

    @patch("builtins.print")
    def test_simulate_mcmc_possession_chain(self, mock_print):
        """Test MCMC possession chain simulation logic."""
        zones = ["Z_2_2", "Z_3_2", "Z_5_2"]
        base_matrix = pd.DataFrame(
            {
                "Z_2_2": [0.0, 0.0, 0.0],
                "Z_3_2": [1.0, 0.0, 0.0],  # Z_2_2 always transitions to Z_3_2
                "Z_5_2": [0.0, 1.0, 0.0],  # Z_3_2 always transitions to Z_5_2 (shot box)
            },
            index=zones,
        )

        df_events = pd.DataFrame({"start_zone": ["Z_2_2", "Z_3_2"], "player": ["Player A", "Player B"]})
        player_profiles = {
            "Player A": {"accuracy": 0.9, "progressive_ratio": 0.5, "total_passes": 10},
            "Player B": {"accuracy": 0.8, "progressive_ratio": 0.4, "total_passes": 10},
        }

        chain = simulate_mcmc_possession_chain("Z_2_2", base_matrix, df_events, player_profiles, zones, max_steps=5)
        self.assertEqual(chain[0], "Z_2_2")
        self.assertEqual(chain[1], "Z_3_2")
        self.assertEqual(chain[2], "Z_5_2")
        # Should stop after entering penalty box (Z_5_2)
        self.assertEqual(len(chain), 3)

    @patch("builtins.print")
    def test_mcmc_corner_scenario_calls_routine_model(self, mock_print):
        """Test that in a corner kick scenario (Z_5_4), the corner routine model is leveraged."""
        zones = ["Z_5_4", "Z_5_2", "Z_4_4"]
        base_matrix = pd.DataFrame(np.eye(3), index=zones, columns=zones)
        df_events = pd.DataFrame({"start_zone": ["Z_5_4"], "player": ["Player A"]})
        player_profiles = {"Player A": {"accuracy": 0.8, "progressive_ratio": 0.3, "total_passes": 10}}

        mock_routine = MagicMock()
        mock_routine.predict_proba.return_value = np.array([[1.0, 0.0, 0.0]])  # Predict Direct Central Box (0)

        mock_outcome = MagicMock()
        mock_outcome.predict_proba.return_value = np.array([0.8])

        chain = simulate_mcmc_possession_chain(
            "Z_5_4", base_matrix, df_events, player_profiles, zones, max_steps=3,
            routine_model=mock_routine, outcome_model=mock_outcome
        )
        mock_routine.predict_proba.assert_called_once()
        self.assertEqual(chain[1], "Z_5_2")

    @patch("builtins.print")
    def test_mcmc_short_corner_bypasses_outcome_model(self, mock_print):
        """Test that for a short corner (routine 2), the outcome model is NOT called and play continues in open play."""
        zones = ["Z_5_4", "Z_4_4", "Z_3_4", "Z_4_0"]
        base_matrix = pd.DataFrame(np.eye(4), index=zones, columns=zones)
        df_events = pd.DataFrame({"start_zone": ["Z_5_4", "Z_4_4"], "player": ["Player A", "Player A"]})
        player_profiles = {"Player A": {"accuracy": 0.8, "progressive_ratio": 0.3, "total_passes": 10}}

        mock_routine = MagicMock()
        mock_routine.predict_proba.return_value = np.array([[0.0, 0.0, 1.0]])  # Predict Short Corner (2)

        mock_outcome = MagicMock()
        mock_outcome.predict_proba.return_value = np.array([0.5])

        chain = simulate_mcmc_possession_chain(
            "Z_5_4", base_matrix, df_events, player_profiles, zones, max_steps=3,
            routine_model=mock_routine, outcome_model=mock_outcome
        )
        mock_routine.predict_proba.assert_called_once()
        mock_outcome.predict_proba.assert_not_called()
        self.assertIn(chain[1], ["Z_4_0", "Z_4_4"])

    @patch("builtins.print")
    def test_mcmc_direct_corner_calls_outcome_model(self, mock_print):
        """Test that for a non-short corner (routine 0 or 1), the outcome model IS called."""
        zones = ["Z_5_4", "Z_5_1", "Z_5_3"]
        base_matrix = pd.DataFrame(np.eye(3), index=zones, columns=zones)
        df_events = pd.DataFrame({"start_zone": ["Z_5_4"], "player": ["Player A"]})
        player_profiles = {"Player A": {"accuracy": 0.8, "progressive_ratio": 0.3, "total_passes": 10}}

        mock_routine = MagicMock()
        mock_routine.predict_proba.return_value = np.array([[0.0, 1.0, 0.0]])  # Predict Post Cross (1)

        mock_outcome = MagicMock()
        mock_outcome.predict_proba.return_value = np.array([0.9])

        chain = simulate_mcmc_possession_chain(
            "Z_5_4", base_matrix, df_events, player_profiles, zones, max_steps=3,
            routine_model=mock_routine, outcome_model=mock_outcome
        )
        mock_routine.predict_proba.assert_called_once()
        mock_outcome.predict_proba.assert_called_once()
        self.assertIn(chain[1], ["Z_5_1", "Z_5_3"])


if __name__ == "__main__":
    unittest.main()
