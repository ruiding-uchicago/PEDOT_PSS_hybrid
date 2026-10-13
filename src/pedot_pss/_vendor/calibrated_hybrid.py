"""Actual GEM plus flexible, explicitly labelled residual calibration.

The caller chooses fit scope (one fold, a platform, or all known data). No
query target or outcome-derived latent inverse enters prediction. Training
reconstruction accuracy is reported separately from any caller-scored CV.
"""
from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

try:
    from .hybrid import _feature_rows, _physics
    from .polybot_features import POLYBOT_KERNEL_FEATURES
except ImportError:
    from .hybrid import _feature_rows, _physics
    from .polybot_features import POLYBOT_KERNEL_FEATURES


SOLVENTS = ("dmso", "eg", "methanol", "ethanol", "ipa", "dmf", "water")
DOSE_HALF_SATURATION = {"volpct": 5.0, "wtpct": 5.0, "mol_L": 0.1}


def _number(row, *names):
    for name in names:
        value = row.get(name)
        if isinstance(value, (numbers.Real, np.number)) and math.isfinite(float(value)):
            return float(value)
    return None


def _duration(row, seconds, minutes):
    value = _number(row, *seconds)
    if value is None:
        value = _number(row, *minutes)
        value = None if value is None else value * 60.0
    return value


def _thermal_stage(temp_C, seconds, activation_energy_eV):
    """Equivalent minutes at 403.15 K; absence is tracked separately."""
    if temp_C is None or seconds is None or seconds < 0 or temp_C <= -273.15:
        return None
    if seconds == 0:
        return 0.0
    log_budget = math.log(seconds / 60.0) - activation_energy_eV / 8.617333262e-5 * (
        1.0 / (temp_C + 273.15) - 1.0 / 403.15)
    return math.exp(float(np.clip(log_budget, -60.0, 30.0)))


def mechanistic_coordinates(feature_dict, activation_energy_eV=0.35):
    """Fixed recipe/process coordinates, with explicit stage and unit boundaries.

    Read already parsed scalar features only. Reported concentrations are
    saturating coordinates, never converted to dry-film GEM fractions.
    """
    row = feature_dict
    out = {}
    stages = {
        "dry": (_number(row, "dry_temperature_C", "pb_first_dry_stage_temperature_C"),
                _duration(row, ("dry_duration_s", "pb_first_dry_duration_s"), ("dry_duration_min",))),
        "anneal": (_number(row, "anneal_temperature_C", "pb_first_anneal_temperature_C"),
                   _duration(row, ("anneal_duration_s", "pb_first_anneal_duration_s"), ("anneal_duration_min",))),
        "post_treatment": (_number(row, "post_treatment_temperature_C"),
                           _duration(row, ("post_treatment_duration_s",), ("post_treatment_duration_min",))),
        "post_dry": (_number(row, "post_temperature_C", "post_treatment_temperature_C", "pb_post_dry_stage_temperature_C", "pb_post_stage_temperature_C"),
                     _number(row, "extra_post_dry_duration__s", "pb_post_dry_duration_s")),
        "second_anneal": (_number(row, "extra_second_anneal_temperature__c", "pb_second_anneal_temperature_C"),
                          _number(row, "extra_second_anneal_duration__s", "pb_second_anneal_duration_s")),
    }
    budgets = []
    for stage, (temperature, duration) in stages.items():
        budget = _thermal_stage(temperature, duration, activation_energy_eV)
        out[f"coord_has_{stage}_thermal_budget"] = float(budget is not None)
        if budget is not None:
            budgets.append(budget)
            out[f"coord_{stage}_thermal_saturation"] = budget / (1.0 + budget)
            out[f"coord_{stage}_thermal_log1p"] = math.log1p(budget)
    total = float(sum(budgets))
    out["coord_has_total_thermal_budget"] = float(bool(budgets))
    out["coord_total_thermal_saturation"] = total / (1.0 + total)
    out["coord_total_thermal_log1p"] = math.log1p(total)

    # Sum solvent saturation coordinates only within the same declared unit.
    stage_saturation = {stage: 0.0 for stage in ("recipe", "post")}
    for stage in ("recipe", "post"):
        for chemical in SOLVENTS:
            for unit, half in DOSE_HALF_SATURATION.items():
                aliases = [f"{stage}_{chemical}_dose_{unit}"]
                if stage == "recipe":
                    aliases.extend([f"{chemical}_dose_{unit}",
                                    f"proc_pre_film_{chemical}_solution_concentration_{unit}"])
                    if unit == "volpct" and chemical in {"dmso", "eg"}:
                        aliases.append(f"pb_{chemical}_volpct_reported")
                else:
                    aliases.extend([f"proc_post_{chemical}_solution_concentration_{unit}",
                                    f"proc_after_deposition_{chemical}_solution_concentration_{unit}"])
                    if unit == "volpct" and chemical in {"methanol", "ethanol", "water"}:
                        aliases.append(f"pb_post_{chemical}_volpct")
                value = _number(row, *aliases)
                name = f"coord_{stage}_{chemical}_saturation_{unit}"
                out["has_" + name] = float(value is not None and value >= 0)
                if value is not None and value >= 0:
                    out[name] = value / (half + value)
                    # Cross-unit sums are avoided: use vol% only in this one
                    # aggregate coordinate and keep other units individually.
                    if unit == "volpct":
                        stage_saturation[stage] += out[name]
        out[f"coord_{stage}_solvent_saturation_volpct_sum"] = stage_saturation[stage]
    recipe, post = stage_saturation["recipe"], stage_saturation["post"]
    thermal = out["coord_total_thermal_saturation"]
    out["coord_recipe_post_solvent_interaction"] = recipe * post
    out["coord_recipe_thermal_interaction"] = recipe * thermal
    out["coord_post_thermal_interaction"] = post * thermal
    dmso = out.get("coord_recipe_dmso_saturation_volpct", 0.0)
    eg = out.get("coord_recipe_eg_saturation_volpct", 0.0)
    out["coord_dmso_eg_interaction"] = dmso * eg
    for key in ("coating_speed_m_s", "post_speed_m_s"):
        value = _number(row, key)
        if value is None:
            pb_value = _number(row, "pb_first_coating_speed_mm_s" if key == "coating_speed_m_s" else "pb_post_speed_mm_s")
            value = None if pb_value is None else pb_value * 0.001
        out[f"coord_has_{key}"] = float(value is not None and value >= 0)
        if value is not None and value >= 0:
            out[f"coord_{key}_log1p_mm_s"] = math.log1p(value / 0.001)
    for key in ("coating_temperature_C", "post_temperature_C"):
        value = _number(row, key, "pb_first_coating_stage_temperature_C" if key == "coating_temperature_C" else "pb_post_stage_temperature_C")
        out[f"coord_has_{key}"] = float(value is not None)
        if value is not None:
            out[f"coord_{key}_centered"] = float(np.clip((value - 100.0) / 100.0, -2.0, 5.0))
    out["coord_solvent_coating_speed_interaction"] = recipe * out.get("coord_coating_speed_m_s_log1p_mm_s", 0.0)
    # Preserve all nine native kernel inputs exactly, without collapsing the
    # post-solvent mixture to one category or an ordinal solvent index.
    for key in POLYBOT_KERNEL_FEATURES:
        value = _number(row, key)
        out["coord_has_" + key] = float(value is not None)
        if value is not None:
            out["coord_" + key] = value
    coverage = _number(row, "structure_coverage_percent")
    out["coord_has_structure_coverage_percent"] = float(coverage is not None)
    if coverage is not None:
        out["coord_structure_coverage_percent"] = coverage
    return out


def _prepare_rows(feature_dicts, activation_energy_eV, feature_mode):
    raw = _feature_rows(feature_dicts)
    physics_rows, residual_rows = [], []
    for row in raw:
        coordinates = mechanistic_coordinates(row, activation_energy_eV)
        physics_row = dict(row)
        # Equivalent exposure at 130 C lets the unchanged 16-coordinate GEM
        # closure use the sum of explicit processing-stage thermal budgets.
        if coordinates["coord_has_total_thermal_budget"]:
            equivalent_minutes = math.expm1(coordinates["coord_total_thermal_log1p"])
            physics_row["anneal_temperature_C"] = 130.0
            physics_row["anneal_duration_s"] = equivalent_minutes * 60.0
        for chemical in ("dmso", "eg"):
            value = _number(row, f"{chemical}_dose_volpct", f"recipe_{chemical}_dose_volpct", f"pb_{chemical}_volpct_reported")
            if value is not None:
                physics_row[f"{chemical}_dose_volpct"] = value
        for key, pb_key, factor in (
            ("coating_speed_m_s", "pb_first_coating_speed_mm_s", .001),
            ("post_speed_m_s", "pb_post_speed_mm_s", .001),
            ("coating_temperature_C", "pb_first_coating_stage_temperature_C", 1.),
            ("post_temperature_C", "pb_post_stage_temperature_C", 1.),
        ):
            if _number(physics_row, key) is None:
                value = _number(row, pb_key)
                if value is not None:
                    physics_row[key] = value * factor
        physics_rows.append(physics_row)
        if feature_mode == "all":
            residual_row = dict(row)
            residual_row.update(coordinates)
        elif feature_mode == "polybot":
            residual_row = {key: value for key, value in coordinates.items()
                            if key.startswith(("coord_pb_", "coord_has_pb_"))
                            or key in {"coord_structure_coverage_percent", "coord_has_structure_coverage_percent"}}
        else:
            residual_row = coordinates
        residual_rows.append(residual_row)
    if physics_rows and feature_mode != "polybot":
        design = _physics.process_design(physics_rows)
        for i, residual_row in enumerate(residual_rows):
            residual_row.update({f"coord_gem_{name}": float(value)
                                 for name, value in zip(_physics.FEATURE_NAMES, design[i])})
    return physics_rows, residual_rows


class _RBFAnchorRidge(RegressorMixin, BaseEstimator):
    """Ridge over target-free training-anchor RBF coordinates."""
    def __init__(self, alpha=1e-4, gamma=0.1, max_centers=256, random_state=20261009):
        self.alpha, self.gamma = alpha, gamma
        self.max_centers, self.random_state = max_centers, random_state

    def fit(self, X, y, sample_weight=None):
        n = X.shape[0]
        if int(self.max_centers) <= 0:
            raise ValueError("max_centers must be positive")
        selected = np.arange(n) if n <= self.max_centers else np.sort(
            np.random.default_rng(self.random_state).choice(n, self.max_centers, replace=False))
        self.centers_ = X[selected].copy()
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=False, solver="lsqr", tol=1e-9)
        self.ridge_.fit(rbf_kernel(X, self.centers_, gamma=self.gamma), y,
                        sample_weight=sample_weight)
        return self

    def predict(self, X):
        check_is_fitted(self, ["centers_", "ridge_"])
        return self.ridge_.predict(rbf_kernel(X, self.centers_, gamma=self.gamma))


class CalibratedGEMRegressor(RegressorMixin, BaseEstimator):
    """Actual GEM plus kernel/tree residual, for caller-declared fit scope.

    alpha must be strictly positive: this class never labels alpha=0 hybrid.
    ``feature_mode='coordinates'`` is useful for small platform domains;
    ``'all'`` additionally retains the supplied scalar/categorical features.
    Exact kernel ridge is guarded above max_kernel_rows; anchor RBF ridge
    limits its basis instead and can be used on larger supplied populations.
    """
    def __init__(self, alpha=1.0, learner="kernel_ridge", learner_params=None,
                 physics_params=None, feature_mode="coordinates", activation_energy_eV=0.35,
                 random_state=20261009, n_jobs=3, max_kernel_rows=2500,
                 fit_scope="caller_supplied_rows"):
        self.alpha = alpha
        self.learner = learner
        self.learner_params = learner_params
        self.physics_params = physics_params
        self.feature_mode = feature_mode
        self.activation_energy_eV = activation_energy_eV
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_kernel_rows = max_kernel_rows
        self.fit_scope = fit_scope

    def _make_learner(self, gamma):
        params = dict(self.learner_params or {})
        if self.learner == "kernel_ridge":
            from sklearn.kernel_ridge import KernelRidge
            defaults = {"alpha": 1e-4, "kernel": "rbf", "gamma": gamma}
            defaults.update(params)
            return KernelRidge(**defaults)
        if self.learner == "rbf_ridge":
            defaults = {"alpha": 1e-4, "gamma": gamma, "max_centers": 256,
                        "random_state": self.random_state}
            defaults.update(params)
            return _RBFAnchorRidge(**defaults)
        if self.learner == "ridge":
            defaults = {"alpha": 1.0, "fit_intercept": False, "solver": "lsqr"}
            defaults.update(params)
            return Ridge(**defaults)
        if self.learner == "extra_trees":
            from sklearn.ensemble import ExtraTreesRegressor
            defaults = dict(n_estimators=500, min_samples_leaf=1, max_features=1.0,
                            random_state=self.random_state, n_jobs=self.n_jobs)
            defaults.update(params)
            return ExtraTreesRegressor(**defaults)
        if self.learner == "catboost":
            from catboost import CatBoostRegressor
            defaults = dict(iterations=800, depth=6, learning_rate=0.04, l2_leaf_reg=1.0,
                            loss_function="RMSE", verbose=False, allow_writing_files=False,
                            random_seed=self.random_state, thread_count=self.n_jobs)
            defaults.update(params)
            return CatBoostRegressor(**defaults)
        if self.learner == "lightgbm":
            from lightgbm import LGBMRegressor
            defaults = dict(n_estimators=800, num_leaves=31, min_child_samples=3,
                            learning_rate=0.04, verbosity=-1, deterministic=True,
                            force_col_wise=True, random_state=self.random_state, n_jobs=self.n_jobs)
            defaults.update(params)
            return LGBMRegressor(**defaults)
        if self.learner == "xgboost":
            from xgboost import XGBRegressor
            defaults = dict(n_estimators=600, max_depth=6, learning_rate=0.04,
                            objective="reg:squarederror", tree_method="hist",
                            reg_lambda=1.0, random_state=self.random_state, n_jobs=self.n_jobs)
            defaults.update(params)
            return XGBRegressor(**defaults)
        raise ValueError("Unsupported learner; choose kernel_ridge, rbf_ridge, ridge, extra_trees, catboost, lightgbm or xgboost")

    def fit(self, feature_dicts, y, sample_weight=None):
        if not isinstance(self.alpha, numbers.Real) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be strictly positive for this actual-hybrid class")
        if self.feature_mode not in {"coordinates", "all", "polybot"}:
            raise ValueError("feature_mode must be coordinates, all or polybot")
        if not math.isfinite(self.activation_energy_eV) or self.activation_energy_eV < 0:
            raise ValueError("activation_energy_eV must be finite and nonnegative")
        physics_rows, rows = _prepare_rows(feature_dicts, self.activation_energy_eV, self.feature_mode)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or not len(rows) or len(rows) != len(target) or not np.all(np.isfinite(target)):
            raise ValueError("features/y must have equal nonzero length and finite log10 targets")
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights is not None and (weights.shape != target.shape or np.any(~np.isfinite(weights))
                                    or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("Invalid sample weights")
        if self.learner == "kernel_ridge" and len(rows) > self.max_kernel_rows:
            raise ValueError("Exact kernel ridge exceeds max_kernel_rows; use rbf_ridge anchors or a tree learner")
        params = dict(max_nfev=150, random_state=self.random_state)
        params.update(self.physics_params or {})
        params["enable_hopping"] = False
        physics = _physics.GreyBoxTransport(**params).fit(physics_rows, target, sample_weight=weights)
        gem = np.asarray(physics.predict(physics_rows))
        vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
        matrix = vectorizer.fit_transform(rows).tocsr()
        # Weighted constant columns can acquire a tiny negative variance from
        # floating-point subtraction inside sklearn; constant scale remains 1.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in sqrt", category=RuntimeWarning,
                                    module=r"sklearn\.preprocessing\._data")
            scaler = StandardScaler(with_mean=False).fit(matrix, sample_weight=weights)
        scaler.var_ = np.maximum(scaler.var_, 0.0)
        scaled = scaler.transform(matrix).tocsr()
        gamma = 1.0 / max(int(np.sum(scaler.var_ > 1e-12)), 1)
        residual_target = target - float(self.alpha) * gem
        intercept = float(np.average(residual_target, weights=weights))
        learner = self._make_learner(gamma)
        fit_kwargs = {"sample_weight": weights} if weights is not None else {}
        safe_names = [f"f{i:06d}" for i in range(matrix.shape[1])]
        if self.learner == "lightgbm":
            fit_kwargs["feature_name"] = safe_names
        learner.fit(scaled, residual_target - intercept, **fit_kwargs)
        self.physics_, self.vectorizer_, self.scaler_ = physics, vectorizer, scaler
        self.residual_model_ = self.model_ = learner
        self.residual_intercept_ = intercept
        self.n_features_in_ = matrix.shape[1]
        self.safe_feature_names_ = safe_names
        self.feature_name_map_ = dict(zip(safe_names, vectorizer.get_feature_names_out()))
        self.default_gamma_ = gamma
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names", category=UserWarning)
            prediction = float(self.alpha) * gem + np.asarray(learner.predict(scaled)) + intercept
        self.fit_info_ = {"n_training_examples": len(rows), "n_encoded_features": matrix.shape[1],
                          "training_reconstruction_rmse_log10": float(np.sqrt(np.mean((prediction-target)**2))),
                          "training_reconstruction_r2_log10": float(r2_score(target,prediction)) if len(rows)>1 else None,
                          "training_metrics_are_resubstitution": True,
                          "gem_equation_residual_max_abs": float(np.max(np.abs(physics.components(physics_rows)["gem_equation_residual"])))}
        return self

    def _query(self, feature_dicts):
        check_is_fitted(self, ["physics_", "vectorizer_", "scaler_", "residual_model_"])
        physics_rows, rows = _prepare_rows(feature_dicts, self.activation_energy_eV, self.feature_mode)
        components = self.physics_.components(physics_rows)
        if not rows:
            return components, np.empty(0)
        matrix = self.scaler_.transform(self.vectorizer_.transform(rows).tocsr())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMRegressor was fitted with feature names", category=UserWarning)
            residual = np.asarray(self.residual_model_.predict(matrix)) + self.residual_intercept_
        return components, residual

    def predict(self, feature_dicts):
        components, residual = self._query(feature_dicts)
        return float(self.alpha) * components["prediction_log10_S_cm"] + residual

    def physics_components(self, feature_dicts):
        components, residual = self._query(feature_dicts)
        gem = components["prediction_log10_S_cm"]
        low = components["effective_low_phase_S_cm"]
        sigma = 10.0 ** gem
        high = self.physics_.sigma_high
        components.update({"physics_prediction_log10_S_cm": gem,
                           "weighted_physics_log10_S_cm": float(self.alpha)*gem,
                           "residual_prediction_log10_S_cm": residual,
                           "combined_prediction_log10_S_cm": float(self.alpha)*gem+residual,
                           "effective_high_phase_S_cm": np.full(len(gem), high),
                           "phase_bounds_ok": (sigma >= np.minimum(low, high)*(1-1e-12)) & (sigma <= np.maximum(low, high)*(1+1e-12))})
        return components

    def summary(self):
        check_is_fitted(self, ["fit_info_"])
        return {"model": "Actual GEM plus flexible residual calibration", "alpha":float(self.alpha),
                "learner":self.learner, "learner_params":self.residual_model_.get_params(),
                "fit_scope":self.fit_scope, "target":"log10 conductivity [S/cm]",
                "feature_mode":self.feature_mode, "activation_energy_eV":self.activation_energy_eV,
                "optional_observed_structure":"structure_coverage_percent is retained when supplied; this makes that caller's task post-characterization.",
                "prediction_equation":"alpha * GEM(process features) + residual_intercept + residual_ML(features)",
                "residual_intercept_log10":self.residual_intercept_, "default_kernel_gamma":self.default_gamma_,
                "physics":self.physics_.summary(), "fit":dict(self.fit_info_),
                "thermal_mapping":"Sum explicit stage Arrhenius budgets; equivalent duration at 403.15 K enters the 16-coordinate GEM closure.",
                "dose_half_saturation":dict(DOSE_HALF_SATURATION),
                "limitations":["Caller determines global/calibration/fold scope and must score these separately.",
                               "Training reconstruction is resubstitution, not held-out validation.",
                               "Thermal activation energy and effective phase geometry are declared priors, not identified microstructure.",
                               "Reported dose saturation is unit/stage specific and is never a dry-film GEM fraction.",
                               "Process interactions enrich the residual; GEM itself retains 16 shared process coefficients.",
                               "Residual flexibility may interpolate known samples without generalizing to new data.",
                               "No query conductivity or outcome-derived inverse latent input is accepted."]}


# Short alias for callers that use the original module naming convention.
CalibratedHybridRegressor = CalibratedGEMRegressor
