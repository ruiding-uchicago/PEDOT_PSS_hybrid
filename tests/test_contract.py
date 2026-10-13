"""Public experiment boundary checks; no training data or fitted model needed."""
import unittest
from pedot_pss import INPUT_FIELDS, normalize


class ExperimentalInputTests(unittest.TestCase):
    def test_unknown_differs_from_zero(self):
        unknown = normalize({"polymer_grade": "PH1000", "dry_duration_min": None})
        zero = normalize({"polymer_grade": "PH1000", "dry_duration_min": 0.})
        self.assertEqual(set(unknown), set(INPUT_FIELDS))
        self.assertIsNone(unknown["dry_duration_min"])
        self.assertEqual(zero["dry_duration_min"], 0.)

    def test_provenance_and_outcomes_are_not_input_fields(self):
        for name in ("doi", "paper_id", "series_label", "sample_id", "coverage_percent",
                     "conductivity_S_cm", "assay_conductivity_sd_S_cm", "thickness_nm"):
            with self.subTest(field=name), self.assertRaises(ValueError):
                normalize({name: 1})

    def test_planned_stages_survive_normalization(self):
        clean = normalize({"process_stages": [{"kind": "dry", "temperature_C": 70.,
                                                "duration_s": 60.}]})
        self.assertEqual(clean["process_stages"][0]["temperature_C"], 70.)
        self.assertEqual(clean["process_stages"][0]["duration_s"], 60.)
        self.assertEqual(normalize(clean), clean)

    def test_invalid_numbers_do_not_silently_become_missing(self):
        for value in (True, float("nan"), float("inf"), "70"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize({"dry_temperature_C": value})
        for field, value in (("dry_duration_min", -1.), ("dry_temperature_C", -273.15),
                             ("stock_solids_wt_percent", 101.)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize({field: value})

    def test_empty_recipe_rejected(self):
        for value in ({}, {field: None for field in INPUT_FIELDS}, {"polymer_grade": "unknown"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize(value)

    def test_additive_pairing_explicit(self):
        with self.assertRaises(ValueError):
            normalize({"additive_names": ["DMSO", "EG"], "additive_amounts_raw": ["4 vol%"]})
        result = normalize({"additive_names": ["DMSO"], "additive_amounts_raw": [None]})
        self.assertEqual(result["additive_names"], ["DMSO"])

    def test_extra_scalar_has_supported_name_number_and_unit(self):
        valid = {"extra_scalar_features": [{"name": "coating_speed", "value": .5, "unit": "mm/s"}]}
        self.assertEqual(normalize(valid)["extra_scalar_features"][0]["value"], .5)
        for item in ({"name": "coating_speed", "value": .5, "unit": "bananas"},
                     {"name": "coverage_percent", "value": .5, "unit": "%"},
                     {"name": "coating_speed", "value": "0.5", "unit": "mm/s"}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                normalize({"extra_scalar_features": [item]})

    def test_stage_invalid_numbers_rejected(self):
        for stage in ({"kind": "dry", "duration_s": -1.},
                      {"kind": "dry", "temperature_C": -273.15},
                      {"kind": "dry", "temperature_C": float("nan")},
                      {"kind": "dry", "duration_range_s": [10., 1.]}):
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                normalize({"process_stages": [stage]})

    def test_new_planned_controls_use_training_units(self):
        inputs = {"extra_scalar_features": [
            {"name": "planned_via_diameter", "value": .002, "unit": "mm"},
            {"name": "measurement_bias", "value": 500., "unit": "mV"},
            {"name": "blend_pedot_pss_dry_mass_percent", "value": 40., "unit": "wt%"},
            {"name": "preparation_water_sodium_mg_l", "value": 33., "unit": "mg/l"},
            {"name": "preparation_water_ph", "value": 7., "unit": "pH"},
        ]}
        clean = normalize(inputs)
        extras = {item["name"]: item for item in clean["extra_scalar_features"]}
        self.assertEqual((extras["planned_via_diameter"]["value"], extras["planned_via_diameter"]["unit"]), (2., "um"))
        self.assertEqual((extras["measurement_bias"]["value"], extras["measurement_bias"]["unit"]), (.5, "V"))
        self.assertEqual(extras["blend_pedot_pss_dry_mass_percent"]["unit"], "%")
        self.assertEqual(extras["preparation_water_sodium_mg_l"]["unit"], "mg/L")
        self.assertEqual(extras["preparation_water_ph"]["unit"], "dimensionless")
        self.assertEqual(inputs["extra_scalar_features"][0]["value"], .002)
        self.assertEqual(normalize(clean), clean)
        for value in (0., -1.):
            with self.assertRaises(ValueError):
                normalize({"extra_scalar_features": [{"name": "planned_via_diameter", "value": value, "unit": "um"}]})

    def test_platform_and_physics_unit_aliases_identify_same_recipe(self):
        values = [
            ("stock_volume", 1000., "ul", 1., "mL"),
            ("mix_duration", 1., "h", 3600., "s"),
            ("mix_rotation_speed", 100., "rev/min", 100., "rpm"),
            ("blade_gap", .2, "mm", 200., "um"),
            ("substrate_uvo_duration", 1., "min", 60., "s"),
            ("post_dry_duration", 1., "min", 60., "s"),
            ("second_anneal_duration", 1., "min", 60., "s"),
            ("post_dry_temperature", 70., "celsius", 70., "C"),
            ("air_storage_time", 0., "day", 0., "s"),
        ]
        for name, value, unit, expected, expected_unit in values:
            with self.subTest(field=name):
                result = normalize({"extra_scalar_features": [{"name": name, "value": value, "unit": unit}]})
                extra = result["extra_scalar_features"][0]
                self.assertEqual(extra["value"], expected)
                self.assertEqual(extra["unit"], expected_unit)


if __name__ == "__main__":
    unittest.main()
