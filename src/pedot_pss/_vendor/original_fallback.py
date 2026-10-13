"""Original M's unknown-source fallback closure, refitted on clean inputs.

This preserves the original global base learners, 17 meta specifications and
global Ridge(.1) fusion. Paper/series expert columns use their actual original
R07 fallback; they do not cause 57 redundant fits. This is the BASE of the
original chain, not a replacement for its later acid/salt/text/MatBERT/P heads.

Only fit() accepts training labels. Prediction accepts canonical experiments
and never opens a manifest, OOF file, training corpus or source-ID lookup.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


from .clean_inputs import canonicalize, original_view, feature_view
from .features import build_features
from .identity_features import enrich_identity_features
from .process_features import enrich_process_features
from .quantity_features import quantity_features
from .reconstruction_features import build_reconstruction_features
from .polybot_features import polybot_features
from .hybrid import GEMResidualRegressor, _physics
from .calibrated_hybrid import _prepare_rows
from .platform_hybrid import PlatformHybrid, platform_mask
from .original_polybot_clean import eligible_scientific

FIVE_COLUMNS = ("prediction_log10", "physics_log10", "weighted_physics_log10",
                "residual_log10", "weighted_domain_correction_log10")
TEACHER_COLUMNS = (0, 2, 3, 4)
_ILLEGAL_PREFIX = ("structure_", "observed_", "context_", "recon_calibration_",
                   "recon_source_", "cv_source_", "cv_label_", "assay_", "pb_historical_",
                   "cv_observed_", "residual_only_assay_")


def _clean_features(row):
    return {k: v for k, v in row.items()
            if not k.startswith(_ILLEGAL_PREFIX) and "coverage" not in k.lower()
            and "historical" not in k.lower()}


def _relative(*args, **kwargs):
    raise RuntimeError("This inference package has no private training specifications; use the complete fitted model")



def load_original_specifications(*args, **kwargs):
    raise RuntimeError("This inference package has no private training specifications; use the complete fitted model")



def _profiles(rawrows):
    """Original feature families after the common experimental projection."""
    canonical = [canonicalize(r) for r in rawrows]
    result = {name: [] for name in ("plain", "identity_quantity", "rich", "state", "enriched")}
    for canonical_row in canonical:
        raw = original_view(canonical_row)
        complete = feature_view(canonical_row)
        planned_extras = {k: v for k, v in complete.items() if k.startswith("extra_")}
        stage_controls = {k: v for k, v in complete.items()
                          if k.startswith(("planned_stage_", "has_planned_stage_"))}
        plain = build_features(raw, include_observed=False, include_context=False)
        plain.update(planned_extras)
        identity_quantity = enrich_identity_features(raw, dict(plain))
        identity_quantity.update(quantity_features(raw))
        rich = enrich_process_features(raw, dict(identity_quantity))
        state = build_reconstruction_features(raw, include_observed=False, include_context=False)
        state.update(polybot_features(raw)); state.update(planned_extras)
        state["pb_platform_supported"] = float(eligible_scientific(complete))
        from .cv_enriched_features import cv_enriched_features
        enriched = dict(state); enriched.update(cv_enriched_features(raw))
        for name, row in (("plain", plain), ("identity_quantity", identity_quantity),
                          ("rich", rich), ("state", state), ("enriched", enriched)):
            row.update(stage_controls)
            result[name].append(_clean_features(row))
    return canonical, result


class _NativeVocabulary:
    """Saved form of global_cv_search.native_matrices' TRAIN-only mapping."""
    def fit(self, rows):
        self.keys = sorted({key for row in rows for key in row})
        self.cat_indices = [j for j, key in enumerate(self.keys)
                            if any(isinstance(row.get(key), str) for row in rows)]
        self.categories = {self.keys[j]: {v: k for k, v in enumerate(sorted({str(row[self.keys[j]])
                            for row in rows if row.get(self.keys[j]) is not None}))}
                           for j in self.cat_indices}
        return self

    def transform(self, rows):
        result = np.empty((len(rows), len(self.keys)), dtype=np.float32)
        for j, key in enumerate(self.keys):
            if key in self.categories:
                mapping = self.categories[key]
                result[:, j] = [mapping.get(str(row[key]), np.nan) if row.get(key) is not None else np.nan for row in rows]
            else:
                result[:, j] = [float(row[key]) if row.get(key) is not None else np.nan for row in rows]
        return result


class _NativeHybrid:
    """Original global native-categorical LightGBM, actual GEM weight .1."""
    def __init__(self, config, seed, n_jobs):
        self.config, self.seed, self.n_jobs = deepcopy(config), seed, n_jobs

    def fit(self, rows, y):
        from lightgbm import LGBMRegressor
        physical, residual = _prepare_rows(rows, .35, "all")
        self.physics = _physics.GreyBoxTransport(max_nfev=120, enable_hopping=False,
                                               random_state=self.seed)
        self.physics.fit(physical, y)
        gem = self.physics.predict(physical)
        specification = self.config["spec"]
        if specification.get("gem_feature"):
            residual = [dict(r, cv_fold_fitted_GEM_log10=float(v)) for r, v in zip(residual, gem)]
        self.vocabulary = _NativeVocabulary().fit(residual)
        self.learner = LGBMRegressor(**specification["params"], random_state=self.seed,
            n_jobs=self.n_jobs, deterministic=True, force_col_wise=True, verbosity=-1)
        self.learner.fit(self.vocabulary.transform(residual), y-specification["alpha"]*gem,
                         categorical_feature=self.vocabulary.cat_indices)
        return self

    def predict(self, rows):
        physical, residual = _prepare_rows(rows, .35, "all")
        gem = self.physics.predict(physical)
        if self.config["spec"].get("gem_feature"):
            residual = [dict(r, cv_fold_fitted_GEM_log10=float(v)) for r, v in zip(residual, gem)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            learned = self.learner.predict(self.vocabulary.transform(residual))
        return self.config["spec"]["alpha"]*gem+learned


def _meta_matrix(candidates, reference):
    return np.column_stack((candidates, reference, candidates.mean(1), candidates.std(1),
                            candidates.min(1), candidates.max(1)))


def _fit_linear(x, y, alpha):
    """Exact original fusion_cv_improve.fit_linear, including sum-loss alpha."""
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-10] = 1.
    z, intercept = (x-mean)/scale, float(y.mean())
    coefficient = np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]), z.T@(y-intercept))
    return {"mean": mean, "scale": scale, "coefficient": coefficient, "intercept": intercept}


def _predict_linear(model, x):
    return ((x-model["mean"])/model["scale"])@model["coefficient"]+model["intercept"]


class _MetaBank:
    def __init__(self, specs, n_jobs, emit):
        self.specs, self.n_jobs, self.emit = deepcopy(specs), n_jobs, emit

    def fit(self, matrix, reference, y, stage):
        self.models, self.alias = {}, {}
        residual = y-reference
        for index, spec in enumerate(self.specs):
            if spec["kind"] == "paper_ridge":
                alpha = spec["alpha"]
                self.alias[index] = next(i for i, s in enumerate(self.specs)
                    if s["kind"] == "ridge" and s["alpha"] == alpha and not s["source_intercepts"])
                continue
            started = time.monotonic()
            self.emit("meta_fit_started", stage=stage, index=index, spec=spec)
            if spec["kind"] == "ridge":
                scaler = StandardScaler().fit(matrix)
                # Identity onehot columns are deleted, not replaced by a fake
                # category. Preserve the original ridge solvers/settings.
                if spec["source_intercepts"]:
                    model = Ridge(alpha=spec["alpha"], solver="lsqr", tol=1e-8)
                    model.fit(sparse.csr_matrix(scaler.transform(matrix)), residual)
                else:
                    model = Ridge(alpha=spec["alpha"])
                    model.fit(scaler.transform(matrix), residual)
                self.models[index] = (scaler, model)
            elif spec["kind"] == "lightgbm":
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(n_estimators=900, learning_rate=.035,
                    num_leaves=spec["leaves"], min_child_samples=spec["child"],
                    reg_lambda=3., verbosity=-1, force_col_wise=True,
                    n_jobs=self.n_jobs, random_state=20261009)
                model.fit(sparse.csr_matrix(matrix), residual)
                self.models[index] = (None, model)
            elif spec["kind"] == "catboost":
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(iterations=900, learning_rate=.04,
                    depth=spec["depth"], l2_leaf_reg=5., thread_count=self.n_jobs,
                    verbose=False, allow_writing_files=False, random_seed=20261009)
                model.fit(matrix, residual)
                self.models[index] = (None, model)
            else:
                raise ValueError("Unknown original meta specification")
            self.emit("meta_fit_complete", stage=stage, index=index,
                      seconds=time.monotonic()-started)
        return self

    def predict(self, matrix, reference):
        prediction = np.empty((len(matrix), len(self.specs)))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            for index, (scaler, learner) in self.models.items():
                x = scaler.transform(matrix) if scaler is not None else matrix
                if self.specs[index]["kind"] == "lightgbm" or self.specs[index].get("source_intercepts"):
                    x = sparse.csr_matrix(x)
                prediction[:, index] = reference+learner.predict(x)
        for index, reference_index in self.alias.items():
            prediction[:, index] = prediction[:, reference_index]
        return prediction


class _BaseBank:
    def __init__(self, specs, outer_fold, n_jobs, polybot_factory, emit):
        self.specs, self.outer_fold, self.n_jobs = specs, outer_fold, n_jobs
        self.polybot_factory, self.emit = polybot_factory, emit

    def fit(self, rawrows, profiles, y, stage):
        self.models = {}
        for entry in self.specs["base_specs"]:
            started = time.monotonic(); position, config = entry["position"], entry["config"]
            self.emit("base_fit_started", stage=stage, position=position, source_config=entry["path"])
            if entry["native"]:
                profile = "state" if config["profile"] == "reconstructed_state" else "enriched"
                model = _NativeHybrid(config, 20261009+self.outer_fold, self.n_jobs)
            else:
                profile = "rich" if config.get("process") else "identity_quantity" if config.get("identity") else "plain"
                model = GEMResidualRegressor(alpha=config["alpha"], ml_kind=config["kind"],
                    model_params=config["params"], random_state=20261009+self.outer_fold,
                    n_jobs=self.n_jobs, physics_params={"max_nfev":config["max_nfev"]})
            model.fit(profiles[profile], y)
            self.models[position] = (profile, model)
            self.emit("base_fit_complete", stage=stage, position=position, seconds=time.monotonic()-started)
        started = time.monotonic(); config = self.specs["r07"]["source_config"]
        local = config["local_spec"] if platform_mask(profiles["state"]).sum() >= 10 else None
        self.emit("base_fit_started", stage=stage, position="R07", source_config="original calibrated ET160 + recipe-gated ET200")
        self.r07 = PlatformHybrid(config["global_spec"], local,
                                  random_state=20261016, n_jobs=self.n_jobs)
        self.r07.fit(profiles["state"], y)
        self.emit("base_fit_complete", stage=stage, position="R07", seconds=time.monotonic()-started)
        self.polybot = None
        if self.polybot_factory is not None:
            native = platform_mask(profiles["state"])
            if native.sum() >= 2:
                self.polybot = self.polybot_factory()
                self.polybot.fit([r for r, use in zip(rawrows, native) if use], y[native])
        return self

    def predict(self, rawrows, profiles):
        r07 = self.r07.physics_components(profiles["state"])
        reference = r07["weighted_physics_log10_S_cm"]
        matrix = np.empty((len(rawrows), self.specs["base_width"]))
        matrix[:, 1:58] = np.asarray(r07["combined_prediction_log10_S_cm"])[:, None]
        for position, (profile, model) in self.models.items():
            matrix[:, position] = model.predict(profiles[profile])
        native = platform_mask(profiles["state"])
        pb_values = None
        if self.polybot is not None and native.any():
            pb_values = np.asarray(self.polybot.predict_values([r for r, use in zip(rawrows, native) if use]), dtype=float)
            if pb_values.shape != (int(native.sum()), 5) or not np.isfinite(pb_values).all():
                raise ValueError("Original cleaned PB override must return aligned five columns")
            if np.max(np.abs(pb_values[:, 0]-pb_values[:, 2]-pb_values[:, 3])) > 1e-10:
                raise ValueError("PB decomposition does not close")
        return matrix, reference, native, pb_values, r07


class OriginalFallbackHybrid(RegressorMixin, BaseEstimator):
    """TRAIN-only original global fallback and learned two-layer fusion.

    Inner splits generate training covariates. All these operations are inside
    the caller's outer TRAIN partition. The inner stacking diagnostics are not
    claimed to be an independently validated inner CV pipeline; only a sealed
    outer-fold query can be used for reporting the complete model's performance.
    """
    def __init__(self, specifications=None, inner_folds=3, inner_seed=20261009,
                 outer_fold=0, n_jobs=3, polybot_factory=None, progress=None):
        self.specifications, self.inner_folds, self.inner_seed = specifications, inner_folds, inner_seed
        self.outer_fold, self.n_jobs = outer_fold, n_jobs
        self.polybot_factory, self.progress = polybot_factory, progress

    def _emit(self, event, **values):
        record = {"event":event, **values}
        if self.progress is None:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        else:
            self.progress(record)

    def fit(self, rawrows, y):
        started = time.monotonic()
        canonical, profiles = _profiles(list(rawrows)); y = np.asarray(y, dtype=float)
        if y.shape != (len(canonical),) or not np.isfinite(y).all() or len(y) < self.inner_folds*2:
            raise ValueError("Finite aligned TRAIN targets and sufficient rows required")
        self.specs_ = deepcopy(self.specifications) if self.specifications is not None else load_original_specifications()
        if self.inner_folds != 3:
            raise ValueError("Predeclared original-clean stack uses exactly three inner folds")
        splitter = KFold(n_splits=3, shuffle=True, random_state=self.inner_seed)
        splits = list(splitter.split(np.arange(len(y))))
        base_oof = np.full((len(y), 67), np.nan); ref_oof = np.full(len(y), np.nan)
        pb_oof = np.full((len(y), 5), np.nan); pb_mask = platform_mask(profiles["state"])
        for inner, (train, query) in enumerate(splits):
            stage = f"inner_base_{inner}"
            selected = {name:[rows[i] for i in train] for name, rows in profiles.items()}
            bank = _BaseBank(self.specs_, self.outer_fold, self.n_jobs, self.polybot_factory, self._emit)
            bank.fit([canonical[i] for i in train], selected, y[train], stage)
            query_profiles = {name:[rows[i] for i in query] for name, rows in profiles.items()}
            matrix, reference, mask, pb_values, _ = bank.predict([canonical[i] for i in query], query_profiles)
            base_oof[query], ref_oof[query] = matrix, reference
            if pb_values is not None:
                pb_oof[query[mask]] = pb_values
            del bank
            self._emit("inner_base_complete", inner_fold=inner, seconds=time.monotonic()-started)
        if not np.isfinite(base_oof).all() or not np.isfinite(ref_oof).all():
            raise ValueError("Incomplete TRAIN base covariates")
        meta_x = _meta_matrix(base_oof[:, self.specs_["meta_positions"]], ref_oof)
        meta_oof = np.full((len(y),17), np.nan)
        for inner, (train, query) in enumerate(splits):
            bank = _MetaBank(self.specs_["meta_specs"], self.n_jobs, self._emit)
            bank.fit(meta_x[train], ref_oof[train], y[train], f"inner_meta_{inner}")
            meta_oof[query] = bank.predict(meta_x[query], ref_oof[query]); del bank
        if self.polybot_factory is not None and pb_mask.any():
            if not np.isfinite(pb_oof[pb_mask]).all():
                raise ValueError("Incomplete clean PB inner-query predictions")
            meta_oof[pb_mask] = pb_oof[pb_mask, 0, None]
        fusion_candidates = self._assemble_fusion(base_oof, meta_oof)
        fusion_x = _meta_matrix(fusion_candidates, ref_oof)
        use = ~pb_mask if self.polybot_factory is not None else np.ones(len(y),dtype=bool)
        if use.sum() < 2:
            raise ValueError("Original global fusion requires at least two non-platform TRAIN rows")
        self.fusion_ = _fit_linear(fusion_x[use], y[use]-ref_oof[use], self.specs_["fusion_alpha"])
        # Late heads need training covariates from this fit's inner QUERY
        # predictions, not the final all-TRAIN fusion resubstitution.
        fusion_oof = np.full(len(y), np.nan)
        for inner, (train, query) in enumerate(splits):
            selected = train[use[train]]
            if len(selected) < 2:
                raise ValueError("Inner fusion requires two non-platform TRAIN rows")
            local = _fit_linear(fusion_x[selected], y[selected]-ref_oof[selected],
                                self.specs_["fusion_alpha"])
            fusion_oof[query] = ref_oof[query]+_predict_linear(local, fusion_x[query])
        self.training_parent_values_ = np.column_stack((fusion_oof, ref_oof, ref_oof,
                                                       fusion_oof-ref_oof, np.zeros(len(y))))
        if self.polybot_factory is not None and pb_mask.any():
            self.training_parent_values_[pb_mask] = pb_oof[pb_mask]
        if (not np.isfinite(self.training_parent_values_).all() or
                np.max(np.abs(self.training_parent_values_[:,0]-self.training_parent_values_[:,2]
                              -self.training_parent_values_[:,3])) > 1e-10):
            raise ValueError("Inner-query training parent does not close")
        self.meta_ = _MetaBank(self.specs_["meta_specs"], self.n_jobs, self._emit)
        self.meta_.fit(meta_x, ref_oof, y, "full_TRAIN_meta")
        self.base_ = _BaseBank(self.specs_, self.outer_fold, self.n_jobs, self.polybot_factory, self._emit)
        self.base_.fit(canonical, profiles, y, "full_TRAIN_base")
        # Remove helper callbacks from fitted banks: serialization must not
        # capture caller logging state or an execution context.
        self.meta_.emit = None; self.base_.emit = None
        self.progress = None
        self.n_training_examples_ = len(y); self.fit_seconds_ = time.monotonic()-started
        self.fold_protocol_ = {"inner_folds":3, "inner_seed":self.inner_seed,
            "outer_QUERY_not_accepted":True, "all_inner_work_confined_to_supplied_TRAIN":True,
            "inner_metrics_not_independent_pipeline_validation":True}
        self._emit("original_fallback_fit_complete", train_n=len(y), seconds=self.fit_seconds_)
        return self

    def _assemble_fusion(self, base, meta):
        if base.shape[1] != 67 or meta.shape[1] != 17:
            raise ValueError("Original base/meta layout changed")
        s = self.specs_["source_fallback"]
        source = s["ridge_weight"]*meta[:,s["meta_ridge_index"]]+s["cat_weight"]*meta[:,s["meta_cat_index"]]
        return np.column_stack((base, meta, source))

    def predict_values(self, rawrows):
        check_is_fitted(self, ["base_", "meta_", "fusion_"])
        canonical, profiles = _profiles(list(rawrows))
        if not canonical:
            return np.empty((0,5))
        base, reference, native, pb_values, _ = self.base_.predict(canonical, profiles)
        meta_x = _meta_matrix(base[:,self.specs_["meta_positions"]], reference)
        meta = self.meta_.predict(meta_x, reference)
        if pb_values is not None:
            meta[native] = pb_values[:,0,None]
        x = _meta_matrix(self._assemble_fusion(base,meta), reference)
        prediction = reference+_predict_linear(self.fusion_,x)
        values = np.column_stack((prediction,reference,reference,prediction-reference,np.zeros(len(prediction))))
        if pb_values is not None:
            values[native] = pb_values
        if not np.isfinite(values).all() or np.max(np.abs(values[:,0]-values[:,2]-values[:,3]))>1e-10:
            raise ValueError("Original fallback prediction/decomposition invalid")
        return values

    def predict(self, rawrows):
        return self.predict_values(rawrows)[:,0]

    def predict_conductivity(self, rawrows):
        return np.power(10.,self.predict(rawrows))

    def summary(self):
        check_is_fitted(self, ["base_", "meta_", "fusion_"])
        return {"model":"original M unknown-source/global-fallback closure, cleaned input refit",
            "train_n":self.n_training_examples_, "fit_seconds":self.fit_seconds_,
            "original_direct_base_learners":len(self.specs_["base_specs"]),
            "R07_global_and_recipe_gated_platform":True,
            "paper_series_columns_using_exact_R07_fallback":57,
            "meta_output_columns":17, "meta_distinct_models":len(self.meta_.models),
            "global_fusion_alpha":self.specs_["fusion_alpha"],
            "source_crossfit_fallback":self.specs_["source_fallback"],
            "five_columns":list(FIVE_COLUMNS), "teacher_columns":list(TEACHER_COLUMNS),
            "protocol":self.fold_protocol_, "no_source_or_outcome_features":True,
            "only_base_closure_not_complete_late_M_or_P":True}
