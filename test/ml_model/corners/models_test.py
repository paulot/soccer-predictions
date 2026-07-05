import unittest
import pandas as pd
import numpy as np
import os
import shutil
from ml_model.corners.models import CornerRoutineXGB, CornerOutcomeXGB


class TestCornerModels(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_temp_corner_models"
        os.makedirs(self.test_dir, exist_ok=True)

        self.X = pd.DataFrame(
            {
                "f1": np.random.rand(50),
                "f2": np.random.rand(50),
            }
        )
        self.y_routine = pd.Series(np.random.choice([0, 1, 2], size=50))
        self.y_outcome = pd.Series(np.random.choice([0, 1], size=50))

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_corner_routine_xgb(self):
        model = CornerRoutineXGB(n_estimators=10)
        model.fit(self.X, self.y_routine)

        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)

        self.assertEqual(len(preds), 50)
        self.assertEqual(probs.shape, (50, 3))

        imp = model.get_feature_importance()
        self.assertIn("f1", imp)

        save_path = os.path.join(self.test_dir, "routine_test.pkl")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded = CornerRoutineXGB.load(save_path)
        loaded_preds = loaded.predict(self.X)
        np.testing.assert_array_equal(preds, loaded_preds)

    def test_corner_outcome_xgb(self):
        model = CornerOutcomeXGB(n_estimators=10)
        model.fit(self.X, self.y_outcome)

        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)

        self.assertEqual(len(preds), 50)
        self.assertEqual(len(probs), 50)

        save_path = os.path.join(self.test_dir, "outcome_test.pkl")
        model.save(save_path)
        self.assertTrue(os.path.exists(save_path))

        loaded = CornerOutcomeXGB.load(save_path)
        loaded_preds = loaded.predict(self.X)
        np.testing.assert_array_equal(preds, loaded_preds)

    def test_corner_model_improvements(self):
        # Test class weights and hyperparameter tuning for Routine model
        r_model = CornerRoutineXGB(n_estimators=5)
        best_params_r = r_model.tune_hyperparameters(
            self.X, self.y_routine, param_grid={"max_depth": [3, 4], "n_estimators": [5]}, cv=2
        )
        self.assertIn("max_depth", best_params_r)
        r_model.fit(self.X, self.y_routine, use_class_weights=True)
        self.assertEqual(len(r_model.predict(self.X)), 50)

        # Test class weights, hyperparameter tuning, and threshold optimization for Outcome model
        o_model = CornerOutcomeXGB(n_estimators=5)
        best_params_o = o_model.tune_hyperparameters(
            self.X, self.y_outcome, param_grid={"max_depth": [3, 4], "n_estimators": [5]}, cv=2
        )
        self.assertIn("max_depth", best_params_o)
        o_model.fit(self.X, self.y_outcome, use_class_weights=True)
        thresh = o_model.optimize_threshold(self.X, self.y_outcome, metric="f1")
        self.assertTrue(0.0 <= thresh <= 1.0)
        self.assertEqual(o_model.threshold, thresh)
        self.assertEqual(len(o_model.predict(self.X)), 50)


if __name__ == "__main__":
    unittest.main()
