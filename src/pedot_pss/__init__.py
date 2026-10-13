"""Experimental PEDOT:PSS recipe in; predicted conductivity in S/cm out."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from collections.abc import Mapping
import math
import os
import sys
import threading

import joblib
import numpy as np

from ._vendor.clean_inputs import INPUT_FIELDS, INPUT_VERSION, canonicalize
from ._validation import validate, require_condition, canonical_extra_units
from ._assets import setup_encoder, setup_model
from ._encoder import shutdown as close_encoder

__version__ = "0.1.0"
_DEFAULT_MODEL = Path(__file__).with_name("models") / "model.joblib"
_MODELS = {}
_LOAD_LOCK = threading.RLock()


def normalize(experiment):
    """Validate public field names and normalize optional experimental inputs."""
    validate(experiment)
    clean = canonicalize(canonical_extra_units(experiment))
    require_condition(clean)
    return clean


@contextmanager
def _trained_module_aliases():
    """Resolve original fitted-class names to the bundled inference code.

    Aliases are needed only while reading the trained file. The loaded objects
    then retain their bundled classes; existing caller modules are restored.
    No private project directory is placed on sys.path.
    """
    aliases = {path.stem: path.stem for path in Path(__file__).with_name("_vendor").glob("*.py")
               if path.stem != "__init__"}
    aliases["_hybrid_expanded_gem_physics"] = "gem_physics"
    prior = {name: sys.modules.get(name) for name in aliases}
    try:
        for name, bundled in aliases.items():
            sys.modules[name] = import_module("pedot_pss._vendor." + bundled)
        yield
    finally:
        for name, module in prior.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _fix_prediction_threads(model):
    """Run the fitted LightGBM and ExtraTrees learners with a fixed thread count (default 1).

    One thread makes the forests' tree sums run in a fixed order, so repeated predictions
    are bitwise identical, and keeps LightGBM's OpenMP runtime from forming thread teams
    (which could otherwise meet PyTorch's runtime when a caller has imported torch).
    Values do not otherwise depend on the thread count. Override with
    PEDOT_PSS_PREDICT_THREADS.
    """
    threads = int(os.environ.get("PEDOT_PSS_PREDICT_THREADS", "1"))
    if threads < 1:
        raise ValueError("PEDOT_PSS_PREDICT_THREADS must be a positive integer")
    from sklearn.ensemble._forest import BaseForest
    try:
        from lightgbm import LGBMModel
    except ImportError:
        LGBMModel = ()
    seen, stack, changed = set(), [model], 0
    while stack:
        obj = stack.pop()
        if id(obj) in seen or isinstance(obj, (str, bytes, int, float, bool, type(None), type,
                                                 np.ndarray, np.generic)):
            continue
        seen.add(id(obj))
        if isinstance(obj, (LGBMModel, BaseForest)):
            obj.set_params(n_jobs=threads)
            changed += 1
            if isinstance(obj, LGBMModel):
                continue
        if isinstance(obj, Mapping):
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())
    return changed


def load_model(model_path=None):
    """Load the complete bundled trained model once per process."""
    configured = model_path or os.environ.get("PEDOT_PSS_MODEL")
    path = Path(configured).expanduser().resolve() if configured is not None else _DEFAULT_MODEL
    with _LOAD_LOCK:
        if path not in _MODELS:
            if not path.is_file():
                raise FileNotFoundError(f"Trained model file is missing: {path}. Run python -m pedot_pss setup-model for the published release.")
            with _trained_module_aliases():
                model = joblib.load(path)
            if not callable(getattr(model, "predict", None)):
                raise TypeError("The trained file must contain a complete predictor")
            # A stacked wrapper (pedot_pss._vendor.stack_infer.StackedSurrogate) carries the fitted
            # predictor, and with it the input version, at .parent.
            version = getattr(model, "input_version_", None)
            if version is None:
                version = getattr(getattr(model, "parent", None), "input_version_", None)
            if version != INPUT_VERSION:
                raise ValueError("The trained model does not match this experimental input version")
            _fix_prediction_threads(model)
            _MODELS[path] = model
        return _MODELS[path]


def predict_many(experiments, *, model_path=None):
    """Predict multiple experimental recipes with one loaded model.

    Returns dictionaries containing conductivity_S_cm and
    log10_conductivity_S_cm. Missing fields become explicit nulls internally.
    """
    clean = [normalize(experiment) for experiment in experiments]
    if not clean:
        return []
    predictions = np.asarray(load_model(model_path).predict(clean), dtype=float)
    if predictions.shape != (len(clean),) or not np.isfinite(predictions).all():
        raise ValueError("The trained model returned invalid predictions")
    result = []
    for log10 in predictions:
        sigma = float(np.power(10., log10))
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Prediction is outside the representable positive conductivity range")
        result.append({"conductivity_S_cm": sigma,
                       "log10_conductivity_S_cm": float(log10)})
    return result


def predict(experiment, *, model_path=None):
    """Predict conductivity from one dictionary of planned experimental inputs."""
    return predict_many([experiment], model_path=model_path)[0]


__all__ = ["predict", "predict_many", "normalize", "load_model", "setup_encoder", "setup_model",
           "close_encoder", "INPUT_FIELDS", "INPUT_VERSION"]
