"""Target-free TRAIN-fitted global anchors and local Matérn-3/2 kernels.

No IO, response/target argument, supervised metric, specimen lookup or query fit.
Caller owns feature admissibility and split membership. Fitted models are plain
dicts of vocabulary, numerical transforms and required TRAIN coordinates (CSR
global centers); no sklearn estimator or row metadata is retained. They are not
JSON-only containers. Numeric covariance deduplication follows the old local
transform's declared 10-decimal standardized-column tolerance, not file hashing.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

VERSION = "joint-kernel-target-free-features-v1"
FLOOR = 1e-12


def _positive(value, name):
    try:
        valid = (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
                 and math.isfinite(float(value)) and value > 0.)
    except (OverflowError, ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Finite positive nonboolean scalar required: " + name)
    return float(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Explicit integer required: " + name)
    return value


def _rows(rows, *, numeric=False, training=False):
    if isinstance(rows, (Mapping, str, bytes)):
        raise ValueError("Iterable of feature dictionaries required, not one mapping/string")
    try:
        rows = list(rows)
    except TypeError as exc:
        raise ValueError("Iterable of feature dictionaries required") from exc
    if training and not rows:
        raise ValueError("Nonempty TRAIN rows required")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Each feature row must be a mapping")
        clean = {}
        for key, value in row.items():
            if type(key) is not str:
                raise ValueError("Strict string feature keys required")
            if type(value) is str:
                if not numeric:
                    clean[key] = value
            elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
                try:
                    number = float(value)
                except (ValueError, OverflowError) as exc:
                    raise ValueError("Representable finite feature required") from exc
                if not math.isfinite(number):
                    raise ValueError("Finite numeric feature required")
                clean[key] = number
            else:
                raise ValueError("Only strings or finite nonboolean numeric features allowed")
        result.append(clean)
    return result


def _vocabulary(vocabulary):
    if (not isinstance(vocabulary, dict) or any(type(k) is not str or type(v) is not int
            for k, v in vocabulary.items()) or sorted(vocabulary.values()) != list(range(len(vocabulary)))):
        raise ValueError("Complete fitted vocabulary required")
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.vocabulary_ = dict(vocabulary)
    vectorizer.feature_names_ = [k for k, _ in sorted(vocabulary.items(), key=lambda p: p[1])]
    return vectorizer


def _encode(vocabulary, rows):
    vectorizer = _vocabulary(vocabulary)
    matrix = (vectorizer.transform(rows) if rows else
              sparse.csr_matrix((0, len(vocabulary)), dtype=float))
    matrix.sum_duplicates()  # Numeric/categorical encoded-name collisions add in the same column.
    return matrix


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("Finite aligned numeric array required: " + name)
    return array


def _indices(value, size, name):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in "iu" or
            np.any(array < 0) or np.any(array >= size) or
            not np.array_equal(array, np.unique(array))):
        raise ValueError("Sorted unique fitted indices required: " + name)
    return array


def _scale2(distance, width):
    positive = distance[distance > 0.]
    median = float(np.median(positive)) if positive.size else 1.
    value = median * width * width
    if not math.isfinite(value):
        raise ValueError("Kernel scale overflow")
    return max(value, FLOOR)


def _sparse_coordinates(vocabulary, scales, rows):
    matrix = _encode(vocabulary, rows).multiply(1. / scales).tocsr()
    if not np.isfinite(matrix.data).all():
        raise ValueError("Scaled sparse feature overflow")
    matrix.data = np.clip(matrix.data, -5., 5.)
    return matrix


def _global_model(model):
    keys = {"version", "kind", "vocabulary", "scales", "centers", "anchor_indices",
            "n_train", "max_centers", "seed", "width", "scale2"}
    if not isinstance(model, dict) or set(model) != keys or model["version"] != VERSION or model["kind"] != "global":
        raise ValueError("Complete global feature model required")
    _vocabulary(model["vocabulary"])
    n = _integer(model["n_train"], "TRAIN count")
    maximum = _integer(model["max_centers"], "max centers")
    _integer(model["seed"], "seed", 0)
    _positive(model["width"], "width"); scale2 = _positive(model["scale2"], "scale2")
    scales = _array(model["scales"], (len(model["vocabulary"]),), "global scales")
    if np.any(scales <= 0.) or scale2 < FLOOR:
        raise ValueError("Positive fitted feature/kernel scales required")
    anchors = _indices(model["anchor_indices"], n, "anchors")
    centers = model["centers"]
    if not sparse.isspmatrix_csr(centers):
        raise ValueError("CSR fitted global centers required")
    centers.check_format(full_check=True)
    if (centers.shape != (min(n, maximum), len(scales)) or len(anchors) != len(centers.indptr)-1 or
            centers.dtype.kind not in "iuf" or not np.isfinite(centers.data).all() or np.any(abs(centers.data) > 5.)):
        raise ValueError("Malformed/clipped fitted center coordinates")
    return scales, centers, scale2


def fit_global(rows, *, max_centers=256, width=1., seed=20261010):
    """Fit TRAIN-only sparse vocabulary/scales/anchors; return (dict, PhiTR)."""
    rows = _rows(rows, training=True)
    maximum = _integer(max_centers, "max centers"); seed = _integer(seed, "seed", 0)
    width = _positive(width, "width")
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(rows)
    matrix.sum_duplicates()
    scales = StandardScaler(with_mean=False).fit(matrix).scale_ if matrix.shape[1] else np.empty(0)
    scales = np.asarray(scales, float)
    if not np.isfinite(scales).all() or np.any(scales <= 0.):
        raise ValueError("Finite positive TRAIN feature scales required")
    matrix = _sparse_coordinates(vectorizer.vocabulary_, scales, rows)
    n, m = len(rows), min(len(rows), maximum)
    anchors = (np.arange(n) if m == n else
               np.sort(np.random.default_rng(seed).choice(n, m, replace=False)))
    centers = matrix[anchors].copy()
    distance = euclidean_distances(centers, centers, squared=True) if matrix.shape[1] else np.zeros((m, m))
    model = dict(version=VERSION, kind="global", vocabulary=dict(vectorizer.vocabulary_),
        scales=scales.copy(), centers=centers, anchor_indices=anchors.copy(), n_train=n,
        max_centers=maximum, seed=seed, width=width, scale2=_scale2(distance, width))
    return model, transform_global(model, rows)


def transform_global(model, queryrows):
    """Transform only; unknown QUERY categories/keys have zero encoded entries."""
    scales, centers, scale2 = _global_model(model)
    rows = _rows(queryrows)
    matrix = _sparse_coordinates(model["vocabulary"], scales, rows)
    distance = (euclidean_distances(matrix, centers, squared=True) if len(rows) and len(scales)
                else np.zeros((len(rows), centers.shape[0])))
    basis = np.exp(-.5 * distance / scale2) / math.sqrt(centers.shape[0])
    return np.column_stack((np.ones(len(rows)), basis))


def _local_coordinates(model, rows):
    if not len(model["selected_indices"]):
        return np.zeros((len(rows), 1))
    matrix = _encode(model["vocabulary"], rows).toarray()[:, model["varying_indices"]]
    matrix = (matrix - model["means"]) / model["scales"]
    if not np.isfinite(matrix).all():
        raise ValueError("Scaled local feature overflow")
    return np.clip(matrix[:, model["selected_indices"]], -5., 5.)


def _local_model(model):
    keys = {"version", "kind", "vocabulary", "means", "scales", "varying_indices",
            "selected_indices", "train_coordinates", "n_train", "width", "scale2", "intercept_variance"}
    if not isinstance(model, dict) or set(model) != keys or model["version"] != VERSION or model["kind"] != "local":
        raise ValueError("Complete local feature model required")
    _vocabulary(model["vocabulary"])
    n = _integer(model["n_train"], "TRAIN count")
    _positive(model["width"], "width"); _positive(model["intercept_variance"], "intercept variance")
    scale2 = _positive(model["scale2"], "scale2")
    varying = _indices(model["varying_indices"], len(model["vocabulary"]), "varying columns")
    selected = _indices(model["selected_indices"], len(varying), "deduplicated columns")
    _array(model["means"], (len(varying),), "TRAIN means")
    scales = _array(model["scales"], (len(varying),), "TRAIN scales")
    coords = _array(model["train_coordinates"], (n, max(1, len(selected))), "local TRAIN coordinates")
    if scale2 < FLOOR or np.any(scales <= 0.) or np.any(abs(coords) > 5.) or (not len(selected) and np.any(coords != 0.)):
        raise ValueError("Malformed local fitted transforms/coordinates")
    return coords, scale2


def _matern(distance, scale2, intercept):
    radius = np.sqrt(3. * np.maximum(distance, 0.) / scale2)
    return intercept + (1. + radius) * np.exp(-radius)


def fit_local(rows, *, width=.8, intercept_variance=1.):
    """Fit numeric TRAIN coordinates and return (plain dict, PSD Ktrain).

    Categories are validated but ignored. Missing numeric keys encode zero as
    DictVectorizer does; this is a feature-container convention, not imputation
    of a physical quantity. Callers must supply availability flags when needed.
    """
    rows = _rows(rows, numeric=True, training=True)
    width = _positive(width, "width"); intercept = _positive(intercept_variance, "intercept variance")
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(rows)
    varying = np.flatnonzero(np.ptp(matrix, axis=0) > 1e-12)
    train = matrix[:, varying]
    if len(varying):
        scaler = StandardScaler().fit(train)
        means, scales = scaler.mean_.copy(), scaler.scale_.copy()
        train = scaler.transform(train)
        if not np.isfinite(train).all():
            raise ValueError("Finite standardized TRAIN coordinates required")
        first = np.argmax(abs(train) > 1e-10, axis=0)
        signs = np.sign(train[first, np.arange(train.shape[1])]); signs[signs == 0.] = 1.
        _, unique = np.unique(np.round((train * signs).T, 10), axis=0, return_index=True)
        selected = np.sort(unique)
        train = np.clip(train[:, selected], -5., 5.)
    else:
        means, scales, selected, train = np.empty(0), np.empty(0), np.empty(0, dtype=int), np.zeros((len(rows), 1))
    distance = euclidean_distances(train, train, squared=True)
    model = dict(version=VERSION, kind="local", vocabulary=dict(vectorizer.vocabulary_),
        means=means, scales=scales, varying_indices=varying, selected_indices=selected,
        train_coordinates=train.copy(), n_train=len(rows), width=width,
        scale2=_scale2(distance, width), intercept_variance=intercept)
    _local_model(model)
    kernel = _matern(distance, model["scale2"], intercept)
    # Canonicalize TRAIN covariance floating-point roundoff only; query is not fitted.
    return model, .5 * kernel + .5 * kernel.T


def query_local(model, queryrows):
    """Return Matérn cross-kernel, shape (n_QUERY, n_TRAIN), without query fit."""
    train, scale2 = _local_model(model)
    rows = _rows(queryrows, numeric=True)
    coords = _local_coordinates(model, rows)
    distance = euclidean_distances(coords, train, squared=True) if rows else np.zeros((0, len(train)))
    return _matern(distance, scale2, model["intercept_variance"])
