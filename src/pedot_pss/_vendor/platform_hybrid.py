"""Recipe-gated platform experts and real-GEM residual ensembles."""
from __future__ import annotations

import copy
import numpy as np

from .hybrid import GEMResidualRegressor


def make_model(spec, random_state=20261009, n_jobs=3):
    spec = copy.deepcopy(spec)
    if float(spec.get("alpha", 1.0)) <= 0:
        raise ValueError("These reconstruction arms require a nonzero physical baseline")
    if spec.get("engine", "legacy") == "calibrated":
        from .calibrated_hybrid import CalibratedGEMRegressor
        return CalibratedGEMRegressor(
            alpha=spec.get("alpha", 1.0), learner=spec["kind"],
            learner_params=spec.get("params", {}),
            feature_mode=spec.get("feature_mode", "all"),
            physics_params={"max_nfev": spec.get("max_nfev", 120)},
            random_state=random_state, n_jobs=n_jobs)
    return GEMResidualRegressor(
        alpha=spec.get("alpha", 1.0), ml_kind=spec["kind"],
        model_params=spec.get("params", {}),
        physics_params={"max_nfev": spec.get("max_nfev", 120)},
        random_state=random_state, n_jobs=n_jobs)


def platform_mask(features):
    return np.asarray([float(f.get("pb_platform_supported", 0.0)) > .5
                       for f in features], bool)


def compact_platform_features(features):
    physical = {"dmso_dose_volpct", "eg_dose_volpct", "anneal_temperature_C",
                "anneal_duration_s", "coating_temperature_C", "coating_speed_m_s",
                "post_temperature_C", "post_speed_m_s", "acid_post_present",
                "polar_solvent_post_present", "reported_pss_pedot_mass_ratio",
                "measurement_temperature_K"}
    return [{key: value for key, value in row.items()
             if key.startswith(("pb_", "structure_", "dry_", "anneal_", "post_treatment_",
                                "extra_second_anneal", "extra_post_dry"))
             or (key.startswith(("recipe_", "post_")) and "dose_" in key)
             or key in physical}
            for row in features]


class PlatformHybrid:
    """Train a full-corpus hybrid and an explicitly recipe-gated lab expert."""

    def __init__(self, global_spec, local_spec=None, *, random_state=20261009, n_jobs=3):
        self.global_spec = global_spec
        self.local_spec = local_spec
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, features, y, sample_weight=None):
        y = np.asarray(y, float)
        self.global_model_ = make_model(self.global_spec, self.random_state, self.n_jobs)
        self.global_model_.fit(features, y, sample_weight=sample_weight)
        local = platform_mask(features)
        self.local_n_ = int(local.sum())
        self.local_model_ = None
        if self.local_spec is not None:
            if self.local_n_ < 10:
                raise ValueError("Platform expert requires at least ten complete recipe rows")
            self.local_model_ = make_model(self.local_spec, self.random_state, self.n_jobs)
            weights = None if sample_weight is None else np.asarray(sample_weight)[local]
            self.local_model_.fit(compact_platform_features(
                [f for f, keep in zip(features, local) if keep]), y[local], sample_weight=weights)
        return self

    def physics_components(self, features):
        parts = dict(self.global_model_.physics_components(features))
        local = platform_mask(features)
        if self.local_model_ is not None and local.any():
            local_parts = self.local_model_.physics_components(compact_platform_features(
                [f for f, keep in zip(features, local) if keep]))
            for key, value in local_parts.items():
                if key in parts:
                    global_value = np.asarray(parts[key])
                    local_value = np.asarray(value)
                    if global_value.shape == (len(features),) and local_value.shape == (int(local.sum()),):
                        merged = global_value.copy()
                        merged[local] = local_value
                        parts[key] = merged
        parts["platform_expert_routed"] = local & (self.local_model_ is not None)
        return parts

    def predict(self, features):
        return np.asarray(self.physics_components(features)["combined_prediction_log10_S_cm"], float)

    def summary(self):
        return {"type": "recipe-gated platform GEM+residual expert",
                "global": self.global_model_.summary(),
                "local": self.local_model_.summary() if self.local_model_ is not None else None,
                "local_training_rows": self.local_n_,
                "routing": "complete reported blade-coating controls and PH1000 platform; no result/row ID gate"}


class PhysicsBlend:
    """Blend trained actual hybrids; each component retains its real GEM solve."""

    def __init__(self, models, weights):
        weights = np.asarray(weights, float)
        if len(models) != len(weights) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid ensemble")
        self.models = models
        self.weights = weights / weights.sum()

    def predict(self, features):
        return sum(w * m.predict(features) for w, m in zip(self.weights, self.models))

    def fit_weights(self, features, y, regularization=.001):
        """Fit nonnegative ensemble calibration on explicitly known labels."""
        from scipy.optimize import minimize
        target = np.asarray(y, float)
        matrix = np.column_stack([model.predict(features) for model in self.models])
        initial = self.weights.copy()
        result = minimize(
            lambda w: float(np.mean((matrix @ w - target) ** 2)
                            + regularization * np.sum((w - initial) ** 2)),
            initial, method="SLSQP", bounds=[(0.0, 1.0)] * len(initial),
            constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
            options={"ftol": 1e-12, "maxiter": 100})
        if not result.success:
            raise ValueError(f"Ensemble calibration failed: {result.message}")
        self.weights = np.maximum(result.x, 0.0)
        self.weights /= self.weights.sum()
        self.weight_fit_ = {"n": len(target), "objective": float(result.fun),
                            "scope": "known-label ensemble calibration, not held-out"}
        return self

    def physics_components(self, features):
        values = [model.physics_components(features) for model in self.models]
        out = {}
        for key in ("combined_prediction_log10_S_cm", "physics_prediction_log10_S_cm",
                    "weighted_physics_log10_S_cm", "residual_prediction_log10_S_cm"):
            out[key] = sum(w * np.asarray(value[key]) for w, value in zip(self.weights, values))
        out["gem_equation_residual"] = np.maximum.reduce([
            np.abs(np.asarray(value["gem_equation_residual"])) for value in values])
        out["phase_bounds_ok"] = np.logical_and.reduce([
            np.asarray(value["phase_bounds_ok"]) for value in values])
        return out

    def summary(self):
        return {"type": "weighted actual-hybrid ensemble",
                "weights": self.weights.tolist(),
                "components": [model.summary() for model in self.models],
                "weight_fit": getattr(self, "weight_fit_", None),
                "physics_boundary": "Each component solves GEM; averaged log predictions are not a single GEM phase solve."}
