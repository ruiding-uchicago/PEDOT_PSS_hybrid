"""Fixed-option (canonical) wording, strict mode, open chemistry and molecules given by SMILES."""
import importlib.util
import json
from pathlib import Path
import unittest
import pedot_pss
from pedot_pss import _categories
from pedot_pss._vendor import features

REPO = Path(__file__).resolve().parents[1]
HAS_RDKIT = importlib.util.find_spec("rdkit") is not None


def log10(result):
    return result["log10_conductivity_S_cm"]


class CanonicalOptionTests(unittest.TestCase):
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

    def test_synonyms_give_identical_predictions(self):
        synonyms = dict(self.recipe, deposition_method="spin-coated (2000 rpm, 60 s)", substrate="glass slide",
                        measurement_method="4-point probe", polymer_grade="Clevios PH1000",
                        additive_names=["dimethyl sulfoxide (DMSO)"])
        result = pedot_pss.predict(synonyms)
        self.assertEqual(log10(pedot_pss.predict(self.recipe)), log10(result))
        self.assertEqual(pedot_pss.normalize(synonyms), pedot_pss.normalize(self.recipe))
        self.assertTrue(any("-> 'DMSO'" in note for note in result["notes"]))

    def test_additive_spellings_are_one_option(self):
        spellings = ["ethylene glycol", "ethylene glycol (EG)", "EG"]
        values = {log10(pedot_pss.predict(dict(self.recipe, additive_names=[name]))) for name in spellings}
        self.assertEqual(len(values), 1)
        self.assertEqual(pedot_pss.normalize(dict(self.recipe, additive_names=["EG"]))["additive_names"],
                         ["ethylene glycol"])

    def test_synonym_groups_from_the_corpus(self):
        self.assertEqual(pedot_pss.normalize(dict(self.recipe, substrate="Mylar"))["substrate"], "PET")
        self.assertEqual(pedot_pss.normalize(dict(self.recipe, polymer_grade="Clevios 1000"))["polymer_grade"], "PH1000")

    def test_strict_accepts_listed_and_rejects_unlisted_options(self):
        synonyms = dict(self.recipe, substrate="glass slide", additive_names=["dimethyl sulfoxide"])
        self.assertEqual(log10(pedot_pss.predict(synonyms, strict=True)), log10(pedot_pss.predict(self.recipe)))
        for change in ({"substrate": "moon rock"}, {"additive_names": ["mystery solvent"]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                pedot_pss.predict(dict(self.recipe, **change), strict=True)

    def test_default_keeps_unlisted_options_with_a_note(self):
        result = pedot_pss.predict(dict(self.recipe, substrate="moon rock"))
        self.assertGreater(result["conductivity_S_cm"], 0.)
        self.assertTrue(any("not a listed option" in note for note in result["notes"]))

    def test_raw_text_mode_uses_text_as_written(self):
        raw = dict(self.recipe, additive_names=["dimethyl sulfoxide"])
        self.assertEqual(pedot_pss.normalize(raw, canonical=False)["additive_names"], ["dimethyl sulfoxide"])
        self.assertNotIn("notes", pedot_pss.predict(raw, canonical=False))

    def test_canonical_wording_is_a_fixed_point(self):
        once = pedot_pss.normalize(dict(self.recipe, substrate="glass slide", additive_names=["EG"]))
        self.assertEqual(pedot_pss.normalize(once), once)


class OpenChemistryTests(unittest.TestCase):
    """Additives and post-treatment chemicals enter through descriptors, not names."""
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((REPO / "examples/planned_spin_coating.json").read_text())

    def test_model_uses_the_open_representation(self):
        from pedot_pss._vendor import openrep
        pedot_pss.load_model()
        status = openrep.status()
        self.assertTrue(status["openrep_installed"])
        self.assertEqual(set(status["modes"].values()), {"D"})

    def test_vocabulary_lists_table_molecules(self):
        vocab = pedot_pss.vocabulary()
        self.assertIn("DMSO", vocab["additive_names"])
        self.assertIn("ethylene glycol", vocab["additive_names"])
        self.assertTrue(any("sulfuric" in name.lower() for name in vocab["post_treatment_chemicals"]))

    def test_unlisted_name_without_structure_is_noted_and_rejected_in_strict_mode(self):
        recipe = dict(self.recipe, additive_names=["mystery solvent"])
        self.assertTrue(any("not a listed option" in n for n in pedot_pss.predict(recipe)["notes"]))
        with self.assertRaises(ValueError):
            pedot_pss.predict(recipe, strict=True)

    def test_identified_chemical_outside_the_additive_table_gets_its_structure(self):
        notes = pedot_pss.predict(dict(self.recipe, additive_names=["toluene"]), strict=True).get("notes", [])
        self.assertFalse(any("not a listed option" in n for n in notes))


@unittest.skipUnless(HAS_RDKIT, "molecules given by SMILES need RDKit (pip install '.[chem]')")
class SmilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipe = json.loads((REPO / "examples/planned_spin_coating.json").read_text())

    def test_listed_molecule_smiles_equals_its_name(self):
        by_name = pedot_pss.predict(self.recipe)
        by_smiles = pedot_pss.predict(dict(self.recipe, additive_names=["solvent A"], additive_smiles=["CS(C)=O"]))
        self.assertEqual(log10(by_name), log10(by_smiles))

    def test_inconsistent_name_and_smiles_rejected(self):
        with self.assertRaises(ValueError):
            pedot_pss.predict(dict(self.recipe, additive_names=["DMSO"], additive_smiles=["CCO"]))

    def test_new_molecules_are_told_apart_and_unregistered_after(self):
        from pedot_pss._vendor import openrep
        before = len(openrep.chem().names)
        molecules = {"ethyl methyl sulfone": "CCS(C)(=O)=O", "ethylene carbonate": "O=C1OCCO1", "hexane": "CCCCCC"}
        values = {}
        for name, smiles in molecules.items():
            result = pedot_pss.predict(dict(self.recipe, additive_names=[name], additive_smiles=[smiles]))
            self.assertTrue(any("enters by its structure" in n for n in result["notes"]))
            values[name] = log10(result)
        self.assertEqual(len(set(values.values())), 3)
        self.assertEqual(len(openrep.chem().names), before)

    def test_post_treatment_chemical_by_smiles(self):
        base = dict(self.recipe, post_treatment="immersed in 1 M Xacid for 10 min; rinsed with DI water",
                    post_treatment_temperature_C=25.0, post_treatment_duration_min=10.0)
        plain = pedot_pss.predict(base)
        acid = pedot_pss.predict(dict(base, post_treatment_smiles={"Xacid": "OS(=O)(=O)C(F)(F)F"}))
        self.assertNotEqual(log10(plain), log10(acid))
        with self.assertRaises(ValueError):
            pedot_pss.predict(dict(base, post_treatment_smiles={"not in the text": "CCO"}))

    def test_smiles_list_must_align(self):
        with self.assertRaises(ValueError):
            pedot_pss.predict(dict(self.recipe, additive_smiles=["CS(C)=O", None]))


if __name__ == "__main__":
    unittest.main()
