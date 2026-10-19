"""Experimental PEDOT:PSS recipe in; predicted conductivity in S/cm out."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from collections.abc import Mapping
import math
import re
import os
import sys
import threading

import joblib
import numpy as np

from ._vendor.clean_inputs import INPUT_FIELDS, INPUT_VERSION, canonicalize
from ._validation import validate, require_condition, canonical_extra_units
from ._assets import setup_encoder, setup_model
from ._encoder import shutdown as close_encoder
from ._canonical import canonical_experiment
from ._categories import unlisted_options, vocabulary
from ._chemistry import merge, registered, resolve_listed_aliases, resolve_structures

__version__ = "0.2.2"
_DEFAULT_MODEL = Path(__file__).with_name("models") / "model.joblib"
_MODELS = {}
_LOAD_LOCK = threading.RLock()


# Temperatures beyond the training data (99.5th percentile) are held at these values by the model, which does not
# represent thermal degradation; substrates have practical limits that a recipe should not exceed.
_SUPPORT_C = {"post_treatment_temperature_C": 200.0, "anneal_temperature_C": 230.0, "dry_temperature_C": 230.0}
_SUBSTRATE_MAX_C = (("pet", 150.0), ("polyethylene terephthalate", 150.0), ("pen", 200.0), ("pdms", 200.0),
                    ("paper", 200.0), ("textile", 180.0), ("fabric", 180.0), ("cotton", 180.0))


_REPEAT = re.compile(r"(?:repeat(?:ed)?|performed|applied|treated|dipped|immersed)[^.;]{0,40}?\b(?:[2-9]|\d{2,}|twice|"
                     r"three|four|five)\s*(?:times|x\b|cycles)|\b(?:[2-9]|\d{2,})\s*(?:times|cycles)\b|\btwice\b|x\s*[2-9]\b")
_NITRIC = re.compile(r"(\d+(?:\.\d+)?)\s*(?:wt\.?\s*%|%)\s*(?:aqueous\s+)?(?:hno3|nitric)|"
                     r"(?:hno3|nitric acid)\s*\(?\s*(\d+(?:\.\d+)?)\s*(?:wt\.?\s*%|%)")


def range_notes(clean):
    """Notes for temperatures beyond the training support or above the substrate's practical limit."""
    notes = []
    temps = {k: clean.get(k) for k in _SUPPORT_C if isinstance(clean.get(k), (int, float))}
    for st in clean.get("process_stages") or []:
        t = st.get("temperature_C") if isinstance(st, Mapping) else None
        if isinstance(t, (int, float)):
            key = "post_treatment_temperature_C" if st.get("kind") in ("post_treat", "rinse") else "anneal_temperature_C"
            temps[key] = max(temps.get(key, t), t)
    for key, t in temps.items():
        if t > _SUPPORT_C[key]:
            notes.append(f"{key} = {t:g} C is above the training data (max ~{_SUPPORT_C[key]:g} C); the model holds "
                         "the prediction at that temperature and does not represent thermal degradation")
    text = " ".join(str(clean.get(k) or "") for k in ("post_treatment", "process_sequence")).lower()
    if _REPEAT.search(text):
        notes.append("a repeat count is written in the text; repeat counts do not change the prediction (each "
                     "treatment is counted once)")
    for m in _NITRIC.finditer(text):
        conc = float(m.group(1) or m.group(2))
        if conc > 70.0:
            notes.append(f"nitric acid at {conc:g} wt% is fuming acid, beyond the ~65-70 wt% used in the literature; "
                         "film damage under such conditions is not modelled")
            break
    substrate = str(clean.get("substrate") or "").lower()
    hottest = max(temps.values(), default=None)
    for word, limit in _SUBSTRATE_MAX_C:
        if hottest is not None and re.search(rf"\b{re.escape(word)}\b", substrate) and hottest > limit:
            notes.append(f"{hottest:g} C exceeds the practical limit of a {word} substrate (~{limit:g} C); the recipe "
                         "is likely not feasible")
            break
    return notes


def _prepare(experiment, strict, canonical=True):
    """(canonical experiment, unlisted molecules to register, notes) for one input dictionary."""
    if not isinstance(experiment, Mapping):
        raise TypeError("An experiment must be a dictionary of experimental conditions")
    experiment, molecules, notes = resolve_structures(experiment)
    validate(experiment)
    experiment = canonical_extra_units(experiment)
    if canonical or strict:
        experiment, rewrites = canonical_experiment(experiment)
        molecules = resolve_listed_aliases(experiment, molecules)
        unlisted = unlisted_options(experiment, registered_names=set(molecules["names"]))
        if strict and unlisted:
            raise ValueError("Options outside the fixed vocabulary (pedot_pss.vocabulary()): "
                             + "; ".join(f"{field}={value!r}" for field, value in unlisted)
                             + ". Use a listed wording, or give an unlisted molecule by SMILES (additive_smiles).")
        notes = notes + rewrites + [f"{field}: {value!r} is not a listed option; used as written" for field, value in unlisted]
    clean = canonicalize(experiment)
    require_condition(clean)
    return clean, molecules, notes + range_notes(clean)


def normalize(experiment, *, canonical=True, strict=False):
    """Validate public field names and normalize optional experimental inputs.

    canonical=True (default) rewrites the categorical fields and additive names to one canonical wording per option
    (the rewrite the training corpus received; see vocabulary()), so synonyms give identical predictions. A rewrite
    is made only if the model's parsers read exactly the same information from it; options outside the vocabulary
    are kept as written. strict=True also rejects options outside the vocabulary. canonical=False uses the text
    exactly as written.
    """
    return _prepare(experiment, strict, canonical)[0]


@contextmanager
def _trained_module_aliases():
    """Resolve original fitted-class names to the bundled inference code.

    Aliases are needed only while reading the trained file. The loaded objects
    then retain their bundled classes; existing caller modules are restored.
    No private project directory is placed on sys.path.
    """
    # chem_descriptors needs RDKit and is imported only for molecules given by SMILES; trained files never name it.
    aliases = {path.stem: path.stem for path in Path(__file__).with_name("_vendor").glob("*.py")
               if path.stem not in ("__init__", "chem_descriptors")}
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


def _install_representation(model):
    """Install the open chemistry representation a model was trained with (descriptors instead of identities).

    The transform is part of the model: the feature families of the base learners and the scientific view of the
    global kernel are built through it, exactly as during training. Models without a stored spec are unchanged.
    """
    spec = (getattr(model, "info", None) or {}).get("openrep_spec")
    if spec:
        from ._vendor import openrep
        openrep.install(openrep.spec_from_payload(spec))


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
            _install_representation(model)
            _MODELS[path] = model
        return _MODELS[path]


def predict_many(experiments, *, model_path=None, canonical=True, strict=False):
    """Predict multiple experimental recipes with one loaded model.

    Returns dictionaries containing conductivity_S_cm and log10_conductivity_S_cm, plus "notes" when an
    input was rewritten to its canonical wording, an option is not in vocabulary(), or an additive was given
    by SMILES. Missing fields become explicit nulls internally.
    """
    prepared = [_prepare(experiment, strict, canonical) for experiment in experiments]
    if not prepared:
        return []
    clean = [item[0] for item in prepared]
    molecules = {"names": {}, "post": {}}
    for item in prepared:
        merge(molecules, item[1])
    model = load_model(model_path)
    with registered(molecules):
        predictions = np.asarray(model.predict(clean), dtype=float)
    if predictions.shape != (len(clean),) or not np.isfinite(predictions).all():
        raise ValueError("The trained model returned invalid predictions")
    result = []
    for log10 in predictions:
        sigma = float(np.power(10., log10))
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Prediction is outside the representable positive conductivity range")
        result.append({"conductivity_S_cm": sigma,
                       "log10_conductivity_S_cm": float(log10)})
    for entry, (_, _, notes) in zip(result, prepared):
        if notes:
            entry["notes"] = list(notes)
    return result


def predict(experiment, *, model_path=None, canonical=True, strict=False):
    """Predict conductivity from one dictionary of planned experimental inputs."""
    return predict_many([experiment], model_path=model_path, canonical=canonical, strict=strict)[0]


__all__ = ["predict", "predict_many", "normalize", "vocabulary", "load_model", "setup_encoder", "setup_model",
           "close_encoder", "INPUT_FIELDS", "INPUT_VERSION"]
