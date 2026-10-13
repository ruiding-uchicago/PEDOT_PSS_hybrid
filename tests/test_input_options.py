"""Strict fixed-option wording and SMILES additive input."""
import importlib.util
import json
from pathlib import Path
import unittest
import pedot_pss
from pedot_pss import _categories, _chemistry
from pedot_pss._vendor import features

REPO = Path(__file__).resolve().parents[1]
HAS_RDKIT = importlib.util.find_spec("rdkit") is not None


class StrictOptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((REPO / "examples/planned_spin_coating.json").read_text())

    def test_canonical_wording_maps_back_to_its_category(self):
        for field, (_, table) in _categories.CATEGORY_FIELDS.items():
            for category, wording in table.items():
                with self.subTest(field=field, category=category):
                    self.assertEqual(_categories.parse(field, wording), [category])
        for key, name in _categories.CHEMICAL_NAMES.items():
            with self.subTest(chemical=key):
                self.assertEqual(features._chemicals(name), [key])

    def test_strict_synonyms_give_identical_predictions(self):
        synonyms = dict(self.recipe, deposition_method="spin-coated (2000 rpm, 60 s)", substrate="glass slide",
                        measurement_method="4-point probe", polymer_grade="Clevios PH1000",
                        additive_names=["dimethyl sulfoxide"])
        a = pedot_pss.predict(self.recipe, strict=True)
        b = pedot_pss.predict(synonyms, strict=True)
        self.assertEqual(a["log10_conductivity_S_cm"], b["log10_conductivity_S_cm"])
        self.assertTrue(b["notes"])

    def test_strict_rejects_unrecognised_options(self):
        for change in ({"substrate": "moon rock"}, {"additive_names": ["mystery solvent"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                pedot_pss.predict(dict(self.recipe, **change), strict=True)

    def test_default_mode_output_has_no_notes(self):
        self.assertNotIn("notes", pedot_pss.predict(self.recipe))


@unittest.skipUnless(HAS_RDKIT, "additive_smiles needs RDKit (pip install '.[chem]')")
class SmilesAdditiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((REPO / "examples/planned_spin_coating.json").read_text())

    def test_listed_molecule_smiles_equals_its_name(self):
        by_name = pedot_pss.predict(self.recipe)
        by_smiles = pedot_pss.predict(dict(self.recipe, additive_names=["solvent A"], additive_smiles=["CS(C)=O"]))
        self.assertEqual(by_name["log10_conductivity_S_cm"], by_smiles["log10_conductivity_S_cm"])

    def test_inconsistent_name_and_smiles_rejected(self):
        with self.assertRaises(ValueError):
            pedot_pss.predict(dict(self.recipe, additive_names=["DMSO"], additive_smiles=["CCO"]))

    def test_unlisted_molecule_enters_by_descriptors_and_is_unregistered_after(self):
        aliases = len(features.CHEMICAL_ALIASES)
        result = pedot_pss.predict(dict(self.recipe, additive_names=["sulfolane"], additive_smiles=["C1CCS(=O)(=O)C1"]))
        self.assertGreater(result["conductivity_S_cm"], 0.)
        self.assertTrue(any("not in the training chemistry" in note for note in result["notes"]))
        self.assertEqual(len(features.CHEMICAL_ALIASES), aliases)
        self.assertIs(features._molecular_registry, _chemistry._features._molecular_registry)

    def test_smiles_list_must_align(self):
        with self.assertRaises(ValueError):
            pedot_pss.predict(dict(self.recipe, additive_smiles=["CS(C)=O", None]))


if __name__ == "__main__":
    unittest.main()
