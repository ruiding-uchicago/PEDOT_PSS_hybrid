"""Actual released-model black-box checks, with no corpus access."""
import json
import math
from pathlib import Path
import unittest
from pedot_pss import predict, predict_many, load_model

REPO = Path(__file__).resolve().parents[1]


class ReleasedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((REPO / "examples/polybot_recipe.json").read_text())

    def test_positive_actual_prediction(self):
        result = predict(self.recipe)
        self.assertTrue(math.isfinite(result["conductivity_S_cm"]))
        self.assertGreater(result["conductivity_S_cm"], 0.)
        self.assertAlmostEqual(math.log10(result["conductivity_S_cm"]),
                               result["log10_conductivity_S_cm"], places=12)

    def test_batch_equals_single_and_model_reused(self):
        one = predict(self.recipe)
        many = predict_many([self.recipe, self.recipe])
        for result in many:
            self.assertAlmostEqual(result["log10_conductivity_S_cm"],
                                   one["log10_conductivity_S_cm"], places=12)
        self.assertIs(load_model(), load_model())

    def test_source_fields_rejected_before_prediction(self):
        for field in ("doi", "paper_id", "conductivity_S_cm"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                predict(dict(self.recipe, **{field: "not-an-input"}))

    def test_text_encoder_runs_in_isolated_torch_only_worker(self):
        import sys
        from pedot_pss import _encoder
        recipe = json.loads((REPO / "examples/planned_spin_coating.json").read_text())
        predict(recipe)
        if _encoder.settings()[0] == "subprocess":
            self.assertNotIn("torch", sys.modules)
            workers = _encoder.worker_info()
            self.assertTrue(workers)
            self.assertEqual(workers[0]["heavy_modules_loaded"], [])
            self.assertEqual(workers[0]["torch_threads"], _encoder.settings()[1])

    def test_new_spin_coating_recipe_uses_complete_general_path(self):
        recipe = json.loads((REPO / "examples/planned_spin_coating.json").read_text())
        result = predict(recipe)
        self.assertTrue(math.isfinite(result["conductivity_S_cm"]))
        self.assertGreater(result["conductivity_S_cm"], 0.)
        self.assertAlmostEqual(math.log10(result["conductivity_S_cm"]),
                               result["log10_conductivity_S_cm"], places=12)


if __name__ == "__main__":
    unittest.main()
