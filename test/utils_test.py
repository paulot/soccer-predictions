import unittest
import numpy as np
import pandas as pd
from utils import parse_location, map_coordinates_to_zone, calculate_brier_score, calculate_log_loss, TEAM_TO_MANAGER


class TestUtils(unittest.TestCase):
    def test_team_to_manager(self):
        """Test that TEAM_TO_MANAGER dictionary is properly defined."""
        self.assertIsInstance(TEAM_TO_MANAGER, dict)
        self.assertIn("Canada", TEAM_TO_MANAGER)
        self.assertEqual(TEAM_TO_MANAGER["Canada"], "John Herdman")

    def test_parse_location_null(self):
        """Test parse_location with null values."""
        self.assertIsNone(parse_location(None))
        self.assertIsNone(parse_location(np.nan))
        self.assertIsNone(parse_location(pd.NA))

    def test_parse_location_list_or_array(self):
        """Test parse_location with list inputs."""
        loc_list1 = [10.0, 20.0]
        self.assertEqual(parse_location(loc_list1), loc_list1)

        loc_list2 = [15.0, 25.0]
        self.assertEqual(parse_location(loc_list2), loc_list2)

    def test_parse_location_string_literal(self):
        """Test parse_location with string representations of lists."""
        self.assertEqual(parse_location("[10.5, 20.5]"), [10.5, 20.5])
        self.assertEqual(parse_location("[10, 20]"), [10.0, 20.0])

    def test_parse_location_invalid(self):
        """Test parse_location with invalid inputs."""
        self.assertIsNone(parse_location("invalid string"))
        self.assertIsNone(parse_location(123))

    def test_map_coordinates_to_zone(self):
        """Test mapping x, y pitch coordinates to discrete zones."""
        # Bottom left corner
        self.assertEqual(map_coordinates_to_zone(0.0, 0.0), "Z_0_0")
        # Middle of pitch
        self.assertEqual(map_coordinates_to_zone(50.0, 40.0), "Z_2_2")
        # Top right corner / out of bounds capping
        self.assertEqual(map_coordinates_to_zone(120.0, 80.0), "Z_5_4")
        self.assertEqual(map_coordinates_to_zone(150.0, 100.0), "Z_5_4")

    def test_calculate_brier_score(self):
        """Test Brier score calculation for match outcomes."""
        # Perfect prediction for Win
        score_perfect = calculate_brier_score(1.0, 0.0, 0.0, "W")
        self.assertAlmostEqual(score_perfect, 0.0)

        # Completely wrong prediction
        score_wrong = calculate_brier_score(0.0, 1.0, 0.0, "W")
        self.assertAlmostEqual(score_wrong, 2.0)

        # Equal probability
        score_equal = calculate_brier_score(0.3333, 0.3333, 0.3334, "D")
        self.assertGreater(score_equal, 0.0)
        self.assertLess(score_equal, 1.0)

    def test_calculate_log_loss(self):
        """Test Log Loss calculation for match outcomes."""
        # High confidence correct prediction
        loss_good = calculate_log_loss(0.95, 0.03, 0.02, "W")
        self.assertAlmostEqual(loss_good, -np.log(0.95))

        # High confidence wrong prediction (clipped at 0.001)
        loss_bad = calculate_log_loss(0.0, 1.0, 0.0, "W")
        self.assertAlmostEqual(loss_bad, -np.log(0.001))

        # Draw outcome
        loss_draw = calculate_log_loss(0.2, 0.6, 0.2, "D")
        self.assertAlmostEqual(loss_draw, -np.log(0.6))


if __name__ == "__main__":
    unittest.main()
