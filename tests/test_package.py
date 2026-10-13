"""Standalone imports and explicit asset setup; no model fit or download."""
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from pedot_pss import _assets, _trained_module_aliases


class StandalonePackageTests(unittest.TestCase):
    def test_original_classes_use_bundled_namespace(self):
        sentinel = types.ModuleType("original_clean_surrogate")
        with patch.dict(sys.modules, {"original_clean_surrogate": sentinel}):
            with _trained_module_aliases():
                module = importlib.import_module("original_clean_surrogate")
                self.assertEqual(module.OriginalCleanSurrogate.__module__,
                                 "pedot_pss._vendor.original_clean_surrogate")
            self.assertIs(sys.modules["original_clean_surrogate"], sentinel)

    def test_physics_platform_branch_uses_bundled_namespace(self):
        with _trained_module_aliases():
            module = importlib.import_module("physics_platform")
            self.assertEqual(module.PhysicsPlatformPolybot.__module__,
                             "pedot_pss._vendor.physics_platform")
        self.assertNotIn("physics_platform", sys.modules)

    def test_encoder_settings_are_explicit(self):
        from pedot_pss import _encoder
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_encoder.settings(), ("subprocess", 1, 900.))
        with patch.dict("os.environ", {"PEDOT_PSS_ENCODER": "inprocess", "PEDOT_PSS_ENCODER_THREADS": "2"}):
            self.assertEqual(_encoder.settings()[:2], ("inprocess", 2))
        for bad in ({"PEDOT_PSS_ENCODER": "gpu"}, {"PEDOT_PSS_ENCODER_THREADS": "0"}):
            with patch.dict("os.environ", bad), self.assertRaises(ValueError):
                _encoder.settings()

    def test_encoder_worker_imports_no_learning_libraries(self):
        source = Path(importlib.import_module("pedot_pss._matbert_worker").__file__).read_text()
        header = source.split("def _disable_optional_transformers_backends", 1)[0]
        for name in ("sklearn", "lightgbm", "catboost", "joblib", "pedot_pss"):
            self.assertNotIn("import " + name, header)

    def test_asset_setup_is_explicit_and_relative(self):
        contents = {"config.json": json.dumps({"hidden_size": 768, "vocab_size": 30522}).encode(),
                    "vocab.txt": b"[PAD]\n", "pytorch_model.bin": b"test-asset"}
        sizes = {name: len(value) for name, value in contents.items()}
        calls = []

        def fetch(request, timeout):
            calls.append(request.full_url)
            self.assertTrue(request.full_url.startswith(_assets.AUTHOR_BASE))
            return io.BytesIO(contents[request.full_url.rsplit("/", 1)[-1]])

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {}, clear=True), patch.object(_assets, "AUTHOR_ASSETS", sizes), patch.object(_assets.urllib.request, "urlopen", fetch):
                location = _assets.setup_encoder(directory)
                self.assertEqual(len(calls), 3)
                self.assertTrue((location / "LICENSE").is_file())
                self.assertEqual(_assets.setup_encoder(directory), location)
                self.assertEqual(len(calls), 3)
                self.assertEqual(_assets.asset_directory("assets/matbert"),
                                 Path(_assets.__file__).resolve().parent / "assets/matbert")

    def test_interrupted_asset_setup_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {}, clear=True), patch.object(_assets, "AUTHOR_ASSETS", {"config.json": 10}), patch.object(_assets.urllib.request, "urlopen", return_value=io.BytesIO(b"short")):
                with self.assertRaises(ValueError):
                    _assets.setup_encoder(directory)
            self.assertFalse(list(Path(directory).glob("*.part")))
            self.assertFalse((Path(directory) / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
