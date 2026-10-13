"""Fold-local GEM prior plus a supervised residual in log10(S/cm).

Only the supplied feature dictionaries and the training targets enter ``fit``.
The previous project's standalone, pure physics module is imported directly;
no data loader, manifest, hash calculation, or training runner is imported.
"""
from __future__ import annotations

import math
import numbers
import re
import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted


from . import gem_physics as _physics


_FORBIDDEN_KEYS = frozenset({
    "metadata", "raw", "raw_row", "row", "targets", "target", "y", "label",
    "source_id", "source_file", "source_path", "experiment_id", "sample_id",
    "sample_label", "doi", "url", "paper_id", "record_id", "row_id",
    "conductivity", "conductivity_s_cm", "conductivity_sm", "log10_conductivity",
    "log10_conductivity_s_cm", "observed_conductivity", "observed_coverage",
    "observed_thickness", "fitted_target_transport_parameters",
})


def _feature_rows(feature_dicts):
    """Copy scalar input features; reject outcome/identity/raw-data containers."""
    rows = []
    for row in feature_dicts:
        if not isinstance(row, Mapping):
            raise TypeError("Each feature row must be a mapping of scalar features")
        clean = {}
        for key, value in row.items():
            if not isinstance(key, str):
                raise TypeError("Feature names must be strings")
            canonical = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if (canonical in _FORBIDDEN_KEYS or canonical.startswith("target_")
                    or canonical.startswith("observed_")
                    or canonical.startswith("metadata_")
                    or canonical.startswith("fitted_target_")):
                raise ValueError(f"Forbidden raw, metadata, or target feature: {key!r}")
            if value is None:
                clean[f"__present__:{key}"] = 0.0
                continue
            if isinstance(value, str):
                present = bool(value.strip())
                clean[f"__present__:{key}"] = float(present)
                if present:
                    clean[key] = value
            elif isinstance(value, (numbers.Real, np.number, np.bool_)):
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TypeError(f"Feature {key!r} must be real-valued") from exc
                present = math.isfinite(number)
                clean[f"__present__:{key}"] = float(present)
                if present:
                    clean[key] = number
            else:
                raise TypeError(f"Feature {key!r} must be a number, string, or None")
        rows.append(clean)
    return rows


class GEMResidualRegressor(RegressorMixin, BaseEstimator):
    """Fit shared GEM physics and a sparse, fold-local residual regressor.

    ``y`` is log10 conductivity in S/cm. The residual target is exactly
    ``y - alpha * GEM(training_features)``. Query prediction takes features
    only and is ``alpha * GEM(query_features) + residual(query_features)``.
    At ``alpha=0`` this is an empirical control, not a hybrid mechanism claim.

    ``model_params`` are passed to the selected residual learner. Additional
    ``physics_params`` can configure the shared GEM closure, but hopping is
    always disabled for this experiment. Every call to ``fit`` creates fresh
    physics, feature-vocabulary, and residual-model components.
    """

    def __init__(self, alpha=0.1, ml_kind="lightgbm", model_params=None,
                 random_state=20261009, n_jobs=3, physics_params=None):
        self.alpha = alpha
        self.ml_kind = ml_kind
        self.model_params = model_params
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.physics_params = physics_params

    def _make_residual_model(self):
        custom = dict(self.model_params or {})
        if self.ml_kind == "lightgbm":
            from lightgbm import LGBMRegressor
            params = dict(n_estimators=450, learning_rate=0.035, num_leaves=23,
                          min_child_samples=8, subsample=0.9, colsample_bytree=0.9,
                          reg_lambda=1.0, verbosity=-1, deterministic=True,
                          force_col_wise=True, random_state=self.random_state,
                          n_jobs=self.n_jobs)
            params.update(custom)
            return LGBMRegressor(**params)
        if self.ml_kind == "catboost":
            from catboost import CatBoostRegressor
            params = dict(iterations=450, learning_rate=0.04, depth=6,
                          l2_leaf_reg=4.0, loss_function="RMSE", verbose=False,
                          allow_writing_files=False, random_seed=self.random_state,
                          thread_count=self.n_jobs, task_type="CPU")
            params.update(custom)
            return CatBoostRegressor(**params)
        if self.ml_kind == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor
            params = dict(n_estimators=400, min_samples_leaf=2, max_features=0.9,
                          random_state=self.random_state, n_jobs=self.n_jobs)
            params.update(custom)
            return ExtraTreesRegressor(**params)
        raise ValueError("ml_kind must be 'lightgbm', 'catboost', or 'extra_trees'")

    def fit(self, feature_dicts, y, sample_weight=None):
        """Fit all components using this call's training rows and targets only."""
        if not isinstance(self.alpha, numbers.Real) or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        rows = _feature_rows(feature_dicts)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(rows) or not len(rows):
            raise ValueError("features/y must have equal nonzero length")
        if not np.all(np.isfinite(target)):
            raise ValueError("Training target must contain only finite log10 S/cm")
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights is not None and (
            weights.shape != target.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or weights.sum() <= 0
        ):
            raise ValueError("Invalid sample weights")

        params = dict(max_nfev=150, random_state=self.random_state)
        params.update(self.physics_params or {})
        params["enable_hopping"] = False
        physics = _physics.GreyBoxTransport(**params)
        physics.fit(rows, target, sample_weight=weights)
        gem_train = np.asarray(physics.predict(rows), dtype=float)
        vectorizer = DictVectorizer(sparse=True, dtype=np.float64, sort=True)
        matrix = vectorizer.fit_transform(rows).tocsr()
        if matrix.shape[1] == 0:
            raise ValueError("At least one scalar input feature is required")
        # Original names may contain JSON control characters or category text.
        # Learners receive only generated names; the vectorizer preserves the
        # human-readable mapping for inspection and serialization.
        safe_names = [f"f{i:06d}" for i in range(matrix.shape[1])]
        residual_model = self._make_residual_model()
        residual_target = target - float(self.alpha) * gem_train
        fit_kwargs = {"sample_weight": weights} if weights is not None else {}
        if self.ml_kind == "lightgbm":
            fit_kwargs["feature_name"] = safe_names
        residual_model.fit(matrix, residual_target, **fit_kwargs)

        components = physics.components(rows)
        sigma = np.power(10.0, gem_train)
        lower = np.minimum(components["effective_low_phase_S_cm"], physics.sigma_high)
        upper = np.maximum(components["effective_low_phase_S_cm"], physics.sigma_high)
        self.physics_ = physics
        self.vectorizer_ = vectorizer
        self.residual_model_ = residual_model
        # Conventional alias is convenient for model inspection in runners.
        self.model_ = residual_model
        self.n_features_in_ = matrix.shape[1]
        self.safe_feature_names_ = safe_names
        self.feature_name_map_ = dict(zip(safe_names, vectorizer.get_feature_names_out()))
        self.fit_info_ = {
            "n_training_examples": len(rows),
            "n_encoded_features": matrix.shape[1],
            "gem_equation_residual_max_abs": float(np.max(np.abs(
                components["gem_equation_residual"]))),
            "gem_phase_bound_violations": int(np.count_nonzero(
                (sigma < lower * (1.0 - 1e-12)) | (sigma > upper * (1.0 + 1e-12)))),
            "training_residual_target_mean_log10": float(np.mean(residual_target)),
            "training_residual_target_std_log10": float(np.std(residual_target)),
        }
        return self

    def _predict_parts(self, rows):
        check_is_fitted(self, ["physics_", "vectorizer_", "residual_model_"])
        gem = np.asarray(self.physics_.predict(rows), dtype=float)
        if not rows:
            return gem, np.empty(0, dtype=float)
        matrix = self.vectorizer_.transform(rows).tocsr()
        # Sparse inputs have no column-label attribute. The fitted vectorizer
        # fixes the column order, and generated names are only learner labels.
        with warnings.catch_warnings():
            if self.ml_kind == "lightgbm":
                warnings.filterwarnings(
                    "ignore", category=UserWarning,
                    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                    module=r"sklearn\.utils\.validation",
                )
            residual = np.asarray(self.residual_model_.predict(matrix), dtype=float)
        return gem, residual

    def predict(self, feature_dicts):
        """Predict from recipe features alone, without query targets."""
        check_is_fitted(self, ["physics_", "vectorizer_", "residual_model_"])
        rows = _feature_rows(feature_dicts)
        gem, residual = self._predict_parts(rows)
        return float(self.alpha) * gem + residual

    def physics_components(self, feature_dicts):
        """Inspect the actual implicit GEM solve and its residual correction.

        Phase-bound checks describe the GEM component. The learned residual's
        final conductivity is not constrained to those effective phase bounds.
        """
        check_is_fitted(self, ["physics_", "vectorizer_", "residual_model_"])
        rows = _feature_rows(feature_dicts)
        components = dict(self.physics_.components(rows))
        gem, residual = self._predict_parts(rows)
        sigma = np.power(10.0, gem)
        low = components["effective_low_phase_S_cm"]
        high = np.full(len(rows), self.physics_.sigma_high)
        lower, upper = np.minimum(low, high), np.maximum(low, high)
        components.update({
            "physics_prediction_log10_S_cm": gem,
            "weighted_physics_log10_S_cm": float(self.alpha) * gem,
            "residual_prediction_log10_S_cm": residual,
            "combined_prediction_log10_S_cm": float(self.alpha) * gem + residual,
            "effective_high_phase_S_cm": high,
            "phase_bounds_ok": (sigma >= lower * (1.0 - 1e-12))
                               & (sigma <= upper * (1.0 + 1e-12)),
        })
        return components

    def summary(self):
        """Return JSON-compatible configuration, fit diagnostics, and limits."""
        check_is_fitted(self, ["physics_", "vectorizer_", "residual_model_"])
        return {
            "model": "empirical residual control" if self.alpha == 0 else "GEM plus ML residual",
            "target": "log10 conductivity [S/cm]",
            "prediction_equation": "alpha * GEM(features) + residual_ML(features)",
            "alpha": float(self.alpha),
            "ml_kind": self.ml_kind,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "model_params": self.residual_model_.get_params(),
            "vectorizer": {"type": "DictVectorizer", "sparse": True,
                           "fit_scope": "training fold only",
                           "availability_indicators": "__present__:<input feature>",
                           "n_features": self.n_features_in_},
            "physics": self.physics_.summary(),
            "fit": dict(self.fit_info_),
            "limitations": [
                "Physics parameters and residual ML are fitted only on supplied training targets.",
                "GEM phases and geometry are effective priors, not identified microstructure.",
                "GEM equation residual and phase bounds verify the solver, not the mechanism.",
                "Training fit diagnostics do not establish held-out accuracy or mechanistic validity.",
                "The residual correction can leave the GEM component's effective phase bounds.",
                "Absent/nonfinite inputs have availability indicators; absence is not measured zero.",
                "Hopping is disabled; no outcome-derived transport inverse enters the features.",
                "An alpha=0 model is an empirical control, not a hybrid model.",
            ],
        }
