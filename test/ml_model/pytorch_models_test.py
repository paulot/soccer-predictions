import unittest
import torch
from ml_model.pytorch_models import OutcomeNN, DestinationNN


class TestPyTorchModels(unittest.TestCase):
    def test_outcome_nn(self):
        """Test OutcomeNN architecture and forward pass."""
        input_dim = 25
        role_idx = 15
        model = OutcomeNN(input_dim=input_dim, role_idx=role_idx)

        # Create dummy batch of 4 samples
        x = torch.randn(4, input_dim)
        # Ensure role index contains valid integers (0 to 3)
        x[:, role_idx] = torch.tensor([0, 1, 2, 3], dtype=torch.float32)

        output = model(x)
        self.assertEqual(output.shape, (4, 1))
        # Ensure output is bounded between 0 and 1 (sigmoid)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_destination_nn(self):
        """Test DestinationNN hybrid attention/MLP architecture and forward pass."""
        input_dim = 70
        role_idx = 10
        def_density_start_idx = 35
        output_dim = 30

        model = DestinationNN(
            input_dim=input_dim, role_idx=role_idx, def_density_start_idx=def_density_start_idx, output_dim=output_dim
        )

        # Check buffer registration
        self.assertIn("static_distances", model._buffers)
        self.assertIn("static_angles", model._buffers)
        self.assertEqual(model.static_distances.shape, (30, 30))

        # Create dummy batch of 4 samples
        x = torch.randn(4, input_dim)
        # Ensure zone coordinates at col 0 and 1 are in valid range
        x[:, 0] = torch.tensor([0, 2, 4, 5], dtype=torch.float32)  # start_zone_x (0 to 5)
        x[:, 1] = torch.tensor([0, 1, 2, 4], dtype=torch.float32)  # start_zone_y (0 to 4)
        x[:, role_idx] = torch.tensor([0, 1, 2, 3], dtype=torch.float32)  # player role

        logits = model(x)
        self.assertEqual(logits.shape, (4, 30))


if __name__ == "__main__":
    unittest.main()
