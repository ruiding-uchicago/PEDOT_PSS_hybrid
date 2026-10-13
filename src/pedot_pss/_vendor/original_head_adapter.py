"""Original selected late-M mathematics, with clean inputs and actual parent states.

Development uses the existing scientific modules. This adapter never calls their
prepare/workflow/replay functions or reads a corpus/OOF file. The original
modules are to be vendored as pure inference components before publication.
Weights are deliberately not applied here: callers select them using TRAIN
only, then compose the predicted delta with the actual preceding model output.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real
import re

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

from . import recipe_text_residual as text
from . import conditional_text_numeric as numeric
from . import nonlinear_text_residual as nonlinear
from . import full_feature_error_views as full_views
from . import full_feature_selected_basis as full_basis
from . import full_feature_error_selected as full_head
from . import char_recipe_neighbor_cv as char
from . import matched_recipe_representation as matched
from . import matbert_signed_selected_basis as material_basis
from .clean_inputs import INPUT_FIELDS

VERSION = "original-selected-late-head-clean-adapter-v1"
KINDS = ("word", "numeric", "nonlinear", "fullfeature", "char", "matbert")
ORIGINAL_WEIGHTS = dict(word=.25, numeric=-1., nonlinear=.25,
    fullfeature=-.3546066790322847, char=.7136382226723879,
    matbert=-.3813368651742054)
_FORBIDDEN = re.compile(r"^(?:paper_id|doi|row_id|source_id|source_file|sample_id|"
    r"sample_label|series_label|batch|conductivity|conductivity_s_cm|target|label)(?:_|$)|"
    r"^(?:structure_|observed_|context_|assay_|cv_|residual_only_|label_|"
    r"paper_|source_|series_|sample_|recon_calibration_)", re.I)


def _rows(rows, name, *, n=None):
    rows = list(rows)
    if n is not None and len(rows) != n:
        raise ValueError(f"{name} rows must match the experimental row count")
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(f"{name} must contain clean experimental dictionaries")
        if any(not isinstance(key, str) or _FORBIDDEN.search(key) for key in row):
            raise ValueError(f"{name} contains prohibited identity, outcome or observed-film fields")
        if name.startswith("raw_") and set(row) - set(INPUT_FIELDS):
            raise ValueError(f"{name} must contain only the public clean experimental fields")
    return rows


def _states(values, n):
    values = np.asarray(values)
    if values.shape != (n, 4) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("parent4 must be finite n×4 actual preceding-model states")
    return np.asarray(values, dtype=float)


def _residual(values, n):
    values = np.asarray(values)
    if values.shape != (n,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("residual_train must be a finite TRAIN-only log-conductivity residual")
    return np.asarray(values, dtype=float)


def _schema(scientific):
    """Same original closed schema filter, from TRAIN keys without corpus lookup."""
    kinds = {}
    for row in scientific:
        for key, value in row.items():
            if value is None:
                continue
            if isinstance(value, str):
                kind = "categorical"
            elif isinstance(value, (Real, np.number, np.bool_)):
                if not np.isfinite(float(value)):
                    raise ValueError("Scientific input values must be finite")
                kind = "numeric"
            else:
                raise TypeError("Scientific features must be scalar numbers, strings or null")
            if key in kinds and kinds[key] != kind:
                raise ValueError("A scientific feature cannot change numeric/category type across TRAIN")
            kinds[key] = kind
    return full_views.make_schema(dict(
        numeric_features=sorted(key for key, kind in kinds.items() if kind == "numeric"),
        categorical_features=sorted(key for key, kind in kinds.items() if kind == "categorical")))


def _full_matrix(basis, scientific, raw, parent4):
    # C1 has no source feature. The legacy helper requires a syntactic source
    # argument, so supply a constant independent of all source/record metadata.
    matrix, _ = full_basis.transform(basis, scientific, raw, parent4,
                                    ["experiment_only"] * len(raw))
    return sparse.csr_matrix(matrix)


def fit_original_head(kind, raw_train, scientific_train, parent4_train,
                      residual_train, *, fold=0, field_vectors_train=None):
    """Fit exactly one originally selected head on supplied clean TRAIN data.

    ``field_vectors_train`` is needed only for MatBERT M and must come from the
    frozen original encoder applied to the same clean experimental text. This
    adapter neither loads nor trains that encoder. No query labels are accepted.
    Returns a complete state dictionary and its TRAIN delta predictions.
    """
    if kind not in KINDS:
        raise ValueError("Unknown original selected head")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold is an internal RNG index in 0..4")
    raw = _rows(raw_train, "raw_train")
    if not raw:
        raise ValueError("Nonempty TRAIN required")
    scientific = _rows(scientific_train, "scientific_train", n=len(raw))
    parent4, residual = _states(parent4_train, len(raw)), _residual(residual_train, len(raw))
    state = dict(version=VERSION, kind=kind, fold=fold, n_train=len(raw),
                 original_weight_for_trace_only=ORIGINAL_WEIGHTS[kind])
    if kind == "word":
        model, _ = text.fit(raw, residual, arm="word", alpha=1)
        state["model"] = model
    elif kind == "numeric":
        basis, designs, _ = numeric.fit_design(raw)
        x = designs["ADDITIVE"]
        estimator = Ridge(**basis["config"]["ridge"]).fit(x, residual)
        state.update(basis=basis, head=dict(coef=np.asarray(estimator.coef_).tolist(),
            intercept=float(estimator.intercept_),
            diagnostics=numeric._diagnostics(x, residual, estimator.coef_, estimator.intercept_)))
    elif kind == "nonlinear":
        basis, x, _ = nonlinear.fit_basis(raw)
        x = nonlinear.append_parent_states(x, parent4)
        state.update(basis=basis,
            head=nonlinear.fit_head(x, residual, fold, float(residual.mean())))
    elif kind == "fullfeature":
        basis, _, _ = full_views.fit_views(scientific, raw, parent4,
            ["experiment_only"] * len(raw), _schema(scientific))
        chosen = full_basis.compact(basis, "C1")
        x = _full_matrix(chosen, scientific, raw, parent4)
        offset = float(residual.mean())
        head, _, _ = full_head.fit_head(x, x[:0], residual, fold, 3, offset)
        state.update(basis=chosen, head=head, offset=offset, capacity_index=3, view="C1")
    elif kind == "char":
        basis, designs, _ = char.fit_basis(raw, parent4, seed=20261011+fold)
        head, _ = char.fit_head(designs["COSINE"], residual)
        state.update(basis=basis, head=head, arm="COSINE")
    else:
        if field_vectors_train is None:
            raise ValueError("MatBERT M requires explicit frozen-encoder TRAIN field vectors")
        basis, x, _ = material_basis.fit_chosen(raw, parent4,
            field_vectors_train, arm="M", fold=fold)
        head, _ = matched.fit_head(x, residual, fold=fold)
        state.update(basis=basis, head=head, arm="M")
    predicted = predict_delta(state, raw, scientific, parent4,
                              field_vectors=field_vectors_train)
    return state, predicted


def predict_delta(state, raw_query, scientific_query, parent4_query,
                  *, field_vectors=None):
    """Pure selected-head delta; no query targets, identifiers or corpus lookup.

    The caller combines this delta using its TRAIN-selected scalar weight.
    The original posthoc coefficients are never silently applied here.
    """
    if not isinstance(state, dict) or state.get("version") != VERSION or state.get("kind") not in KINDS:
        raise ValueError("A complete original-head adapter state is required")
    raw = _rows(raw_query, "raw_query")
    scientific = _rows(scientific_query, "scientific_query", n=len(raw))
    parent4 = _states(parent4_query, len(raw))
    kind = state["kind"]
    if kind == "word":
        result = text.predict(state["model"], raw)
    elif kind == "numeric":
        numeric._validate(state["basis"], heads=False)
        values, known, audit = numeric._project(raw)
        designs = numeric._design(state["basis"], values, known, audit)
        result = np.asarray(designs["ADDITIVE"] @ np.asarray(state["head"]["coef"])).ravel()
        result += state["head"]["intercept"]
    elif kind == "nonlinear":
        x, _ = nonlinear.transform_basis(state["basis"], raw)
        x = nonlinear.append_parent_states(x, parent4)
        result = nonlinear.predict_head(state["head"], x)
    elif kind == "fullfeature":
        x = _full_matrix(state["basis"], scientific, raw, parent4)
        booster = lgb.Booster(model_str=state["head"]["model_string"])
        _, result = full_head.predictions(booster, x[:0], x, state["offset"])
    elif kind == "char":
        designs, _ = char.transform_basis(state["basis"], raw, parent4)
        result = char.predict_head(state["head"], designs["COSINE"])
    else:
        if field_vectors is None:
            raise ValueError("MatBERT M requires frozen-encoder QUERY field vectors")
        x, _ = material_basis.transform(state["basis"], raw, parent4, field_vectors)
        result = matched.predict_head(state["head"], x)
    result = np.asarray(result, dtype=float)
    if result.shape != (len(raw),) or not np.isfinite(result).all():
        raise ValueError("Incomplete or nonfinite query delta prediction")
    return result


def copy_state(state):
    """Complete reusable fitted state; no raw TRAIN records or labels added."""
    return deepcopy(state)
