"""TRAIN-only nonlinear recipe error heads; plain native strings, no file IO.

These are ML residual corrections, not new physics. Caller owns membership,
NONNESTED teacher provenance, and the seventeen protected physical conditions.
QUERY has no label argument. All saved trees and the explicit TRAIN mean offset
are replayed; there is no external init_score, init_model, or hidden target mean.
"""
from copy import deepcopy
from importlib.metadata import version as package_version
from numbers import Real
import math
import re

import lightgbm as lgb
import numpy as np
from scipy import sparse

from . import conditional_text_numeric as numeric
VERSION = "nonlinear-field-text-numeric-error-head-v1"
ARMS = ("T0", "T1")
STATE_COLUMNS = (0, 2, 3, 4)
_VERSIONS = {k: package_version(k) for k in ("lightgbm", "scikit-learn", "numpy", "scipy")}
_HEAD_KEYS = {"version", "config", "fold", "n_train", "n_features", "offset",
              "actual_trees", "effective_parameters", "model_string", "diagnostics"}
_DIAG_KEYS = {"n_rows", "n_features", "residual_mean", "centered_mean", "sse",
              "mean_baseline_sse", "prediction_min", "prediction_max"}


def configuration():
    """JSON-safe fixed configuration; installed versions are part of replay."""
    if _VERSIONS["lightgbm"] != "4.6.0":
        raise ValueError("Reviewed LightGBM 4.6.0 required")
    learner = dict(objective="regression", n_estimators=900, learning_rate=.035,
        num_leaves=31, max_depth=-1, min_child_samples=8, min_child_weight=.001,
        reg_alpha=0., reg_lambda=3., min_split_gain=0., max_bin=63,
        min_data_in_bin=3, subsample=1., subsample_freq=0, colsample_bytree=1.,
        boost_from_average=False, zero_as_missing=False, deterministic=True,
        force_col_wise=True, n_jobs=2, verbosity=-1)
    return dict(version=VERSION, arms=list(ARMS), basis=numeric.configuration(),
        state_columns=list(STATE_COLUMNS), learner=learner, seed_base=20261009,
        versions=dict(_VERSIONS), offset="explicit TRAIN residual mean; no init_score",
        prediction="offset + all native trees raw_score; ML-only NONNESTED teacher")


def _scalar(value):
    try:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real): raise ValueError()
        value = float(value)
        if not math.isfinite(value): raise ValueError()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Finite nonboolean scalar required") from exc
    return value


def _array(values, shape):
    a = np.asarray(values)
    if (a.shape != shape or a.dtype.kind not in "iuf" or not np.isfinite(a).all()
            or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(values, dtype=object).flat)):
        raise ValueError("Finite aligned nonboolean numeric array required")
    return a.astype(np.float64, copy=True)


def _matrix(matrix, *, training=False):
    if sparse.issparse(matrix):
        if matrix.ndim != 2 or matrix.dtype.kind not in "iuf" or not np.isfinite(matrix.data).all():
            raise ValueError("Finite nonboolean numeric sparse matrix required")
        result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    else:
        a = np.asarray(matrix)
        if a.ndim != 2: raise ValueError("Two-dimensional feature matrix required")
        result = sparse.csr_matrix(_array(matrix, a.shape))
    if not 1 <= result.shape[1] <= 6014 or (training and not result.shape[0]):
        raise ValueError("Nonempty TRAIN and 1..6014 feature columns required")
    result.sum_duplicates(); result.sort_indices(); result.eliminate_zeros()
    if not np.isfinite(result.data).all(): raise ValueError("Sparse aggregation overflow")
    return result


def fit_basis(raw_train):
    """One unsupervised TRAIN vocabulary/scaler fit, with no Ridge heads."""
    basis, designs, audit = numeric.fit_design(raw_train)
    numeric._validate(basis, heads=False)
    return deepcopy(basis), _matrix(designs["ADDITIVE"], training=True), audit


def transform_basis(basis, raw_query):
    """Use saved TRAIN state only; unknown text/numeric never triggers a fit."""
    numeric._validate(basis, heads=False)
    values, known, audit = numeric._project(raw_query)
    return _matrix(numeric._design(basis, values, known, audit)["ADDITIVE"]), audit


def append_parent_states(T0, finite_n_by4):
    """Caller supplies parent columns[0,2,3,4], never query targets/identities."""
    matrix = _matrix(T0)
    if matrix.shape[1] > 6010: raise ValueError("T0 dimension exceeds 6010")
    states = _array(finite_n_by4, (matrix.shape[0], 4))
    return _matrix(sparse.hstack((matrix, sparse.csr_matrix(states)), format="csr"))


def _fold(fold):
    if type(fold) is not int or not 0 <= fold < 5: raise ValueError("Integer fold0..4 required")
    return fold


def _native(booster, fold):
    """Check actual native parameters, not only a saved requested config."""
    params = booster._get_loaded_param()  # reviewed LightGBM 4.6.0 native JSON
    wanted = dict(boosting="gbdt", objective="regression", num_iterations=900,
        learning_rate=.035, num_leaves=31, max_depth=-1, min_data_in_leaf=8,
        min_sum_hessian_in_leaf=.001, lambda_l1=0., lambda_l2=3., min_gain_to_split=0.,
        max_bin=63, min_data_in_bin=3, bagging_fraction=1., bagging_freq=0,
        feature_fraction=1., boost_from_average=False, zero_as_missing=False,
        deterministic=True, force_col_wise=True, num_threads=2, verbosity=-1,
        seed=20261009+fold)
    if any(k not in params or type(params[k]) is bool and type(v) is not bool
           or params[k] != v for k, v in wanted.items()):
        raise ValueError("Native model parameters do not match fixed configuration")
    return params


def _diagnostics(X, y, prediction, offset):
    with np.errstate(over="raise", invalid="raise"):
        try:
            error = prediction-y; centered = y-offset
            result = dict(n_rows=len(y), n_features=X.shape[1], residual_mean=float(y.mean()),
                centered_mean=float(centered.mean()), sse=float(error@error),
                mean_baseline_sse=float(centered@centered),
                prediction_min=float(prediction.min()), prediction_max=float(prediction.max()))
        except FloatingPointError as exc:
            raise ValueError("Finite TRAIN diagnostics required") from exc
    if any(not math.isfinite(v) for v in result.values()): raise ValueError("Finite TRAIN diagnostics required")
    return result


def _load(head):
    if (type(head) is not dict or head.keys() != _HEAD_KEYS or head["version"] != VERSION
            or not numeric.text._same(head["config"], configuration())):
        raise ValueError("Closed head schema/configuration required")
    fold = _fold(head["fold"])
    if (type(head["n_train"]) is not int or head["n_train"] < 1
            or type(head["n_features"]) is not int or not 1 <= head["n_features"] <= 6014
            or type(head["actual_trees"]) is not int or not 1 <= head["actual_trees"] <= 900
            or type(head["offset"]) is not float):
        raise ValueError("Closed positive dimensions, actual tree count and float offset required")
    _scalar(head["offset"])
    d = head["diagnostics"]
    if (type(d) is not dict or d.keys() != _DIAG_KEYS
            or type(d["n_rows"]) is not int or d["n_rows"] != head["n_train"]
            or type(d["n_features"]) is not int or d["n_features"] != head["n_features"]
            or any(type(d[k]) is not float or not math.isfinite(d[k]) for k in _DIAG_KEYS-{"n_rows","n_features"})
            or d["sse"] < 0. or d["mean_baseline_sse"] < 0.
            or d["prediction_min"] > d["prediction_max"] or d["residual_mean"] != head["offset"]):
        raise ValueError("Closed finite TRAIN diagnostics required")
    model = head["model_string"]
    if (type(model) is not str or len(model) > 32*1024**2 or "\x00" in model or not model.startswith("tree\nversion=v4\n")
            or "\nend of trees\n" not in model or "\nparameters:\n" not in model
            or re.search(r"(?i)(?<![a-z])(?:nan|[-+]?inf(?:inity)?)(?![a-z])", model)):
        raise ValueError("Finite complete native tree string required")
    try:
        booster = lgb.Booster(model_str=model)
        dumped = booster.dump_model(num_iteration=-1)
        if (dumped["objective"] != "regression" or dumped["num_class"] != 1 or dumped["average_output"]
                or booster.num_feature() != head["n_features"] or booster.num_trees() != head["actual_trees"]
                or booster.current_iteration() != head["actual_trees"] or booster.num_model_per_iteration() != 1
                or booster.feature_name() != [f"f{i}" for i in range(head["n_features"])]
                or not numeric.text._same(_native(booster, fold), head["effective_parameters"])):
            raise ValueError("Native tree dimensions/count/features/effective parameters changed")
    except lgb.basic.LightGBMError as exc:
        raise ValueError("Invalid native tree model") from exc
    return booster


def restore_head(head):
    """Validate/copy a plain saved head; native string load, never fit or file IO."""
    _load(head)
    return deepcopy(head)


def predict_head(head, X):
    """All native trees raw_score plus the same stored offset; no QUERY labels."""
    booster = _load(head); matrix = _matrix(X)
    if matrix.shape[1] != head["n_features"]: raise ValueError("Saved feature width changed")
    if not matrix.shape[0]: return np.empty(0, dtype=np.float64)
    prediction = np.asarray(booster.predict(matrix, raw_score=True, start_iteration=0,
        num_iteration=-1, num_threads=2), dtype=np.float64)+head["offset"]
    if prediction.shape != (matrix.shape[0],) or not np.isfinite(prediction).all():
        raise ValueError("Finite aligned full-tree prediction required")
    return prediction


def fit_head(X, residual_train, fold, offset):
    """One supervised TRAIN fit; explicit offset must equal the TRAIN mean."""
    fold = _fold(fold); matrix = _matrix(X, training=True)
    y = _array(residual_train, (matrix.shape[0],)); offset = _scalar(offset)
    with np.errstate(over="raise", invalid="raise"):
        try:
            if offset != float(y.mean()): raise ValueError("Offset must equal TRAIN residual mean")
            centered = y-offset
        except FloatingPointError as exc: raise ValueError("Finite centered TRAIN labels required") from exc
    if np.max(abs(centered), initial=0.) > np.finfo(np.float32).max:
        raise ValueError("Centered native float32 labels not representable")
    estimator = lgb.LGBMRegressor(**configuration()["learner"], random_state=20261009+fold)
    estimator.fit(matrix, centered, feature_name=[f"f{i}" for i in range(matrix.shape[1])])
    model = estimator.booster_.model_to_string(num_iteration=-1, start_iteration=0)
    booster = lgb.Booster(model_str=model)
    prediction = np.asarray(booster.predict(matrix, raw_score=True, start_iteration=0,
        num_iteration=-1, num_threads=2), dtype=np.float64)+offset
    head = dict(version=VERSION, config=configuration(), fold=fold, n_train=len(y),
        n_features=matrix.shape[1], offset=offset, actual_trees=booster.num_trees(),
        effective_parameters=_native(booster, fold), model_string=model,
        diagnostics=_diagnostics(matrix,y,prediction,offset))
    return restore_head(head)


def replay_train(head, X, residual_train):
    """No-fit saved-tree predictions and independent double TRAIN diagnostics."""
    if type(head) is not dict: raise ValueError("Closed head mapping required")
    matrix = _matrix(X, training=True); y = _array(residual_train,(matrix.shape[0],))
    if len(y) != head.get("n_train"): raise ValueError("TRAIN membership count changed")
    prediction = predict_head(head,matrix)
    diagnostics = _diagnostics(matrix,y,prediction,head["offset"])
    error = max(abs(diagnostics[k]-head["diagnostics"][k]) for k in _DIAG_KEYS)
    if diagnostics["residual_mean"] != head["offset"] or error > 1e-9:
        raise ValueError("Saved TRAIN labels/design/diagnostics changed")
    return dict(predictions=prediction, diagnostics=diagnostics, maximum_diagnostic_error=float(error))
