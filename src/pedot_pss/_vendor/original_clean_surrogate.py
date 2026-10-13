"""Original source-free M path and physical P, as a complete recipe predictor.

All learned upstream components are fitted inside the caller's TRAIN set.
Inner cross-fitting supplies residual-learning inputs; the caller's outer
QUERY labels are never accepted by any prediction function. Raw provenance,
same-assay statistics and observed finished-film properties are projected out.
"""
from __future__ import annotations
import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted

from .clean_inputs import INPUT_VERSION, canonicalize, original_view, feature_view
from .original_fallback import OriginalFallbackHybrid
from .original_polybot_clean import OriginalPolybotClean, eligible_scientific
from .original_global_kernel import OriginalGlobalKernel
from .original_head_adapter import fit_original_head, predict_delta
from .original_physics_adapter import fit_original_physics, predict_original_physics
from .offline_matbert_fields import FrozenMatBERT

VERSION = "original-structure-experiment-only-MP-1012"
PARENT4 = (0, 2, 3, 4)
LATE_HEADS = ("word", "numeric", "nonlinear", "fullfeature", "char", "matbert")


def _values(result):
    return np.asarray(result[0] if isinstance(result, tuple) else result, dtype=float)


def _closure(values):
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("Finite complete model five-column state required")
    if np.max(abs(values[:, 0]-values[:, 2]-values[:, 3]), initial=0.) > 1e-9:
        raise ValueError("Model physical/residual decomposition did not close")


def _slice(rows, indices):
    return [rows[i] for i in indices]


class OriginalCleanSurrogate(RegressorMixin, BaseEstimator):
    """Unified experimental inputs → M or physics-preferred P conductivity.

    The original model families and numerical equations are retained. Source
    local routes use their original global defaults; source bias and observed
    references are removed. Late scalar coefficients are learned on TRAIN
    internal cross-fitted deltas, not copied from the retrospective selection.
    """
    def __init__(self, specifications=None, outer_fold=0, n_jobs=3,
                 special_factory=None, filler_factory=None,
                 encoder_asset_directory=None, prefer_physics=True, progress=None):
        self.specifications, self.outer_fold, self.n_jobs = specifications, outer_fold, n_jobs
        self.special_factory, self.filler_factory = special_factory, filler_factory
        self.encoder_asset_directory, self.prefer_physics = encoder_asset_directory, prefer_physics
        self.progress = progress

    def _emit(self, event, **data):
        if self.progress:
            self.progress({"event": event, "outer_fold": self.outer_fold, **data})

    @staticmethod
    def _project(X, scientific=None):
        clean = [canonicalize(row) for row in X]
        raw = [original_view(row) for row in clean]
        science = [feature_view(row) for row in clean] if scientific is None else list(scientific)
        if len(science) != len(raw):
            raise ValueError("Scientific rows must correspond exactly to experiments")
        active = np.asarray([not eligible_scientific(row) for row in science], dtype=bool)
        return clean, raw, science, active

    def fit(self, X, y, *, field_vectors=None, scientific=None, prefitted_base=None):
        started = time.monotonic()
        clean, raw, science, active = self._project(list(X), scientific)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(raw),) or not len(raw) or not np.isfinite(y).all():
            raise ValueError("Finite matching TRAIN log10(S/cm) targets required")
        self.input_version_ = INPUT_VERSION
        encoder = FrozenMatBERT(self.encoder_asset_directory)
        if field_vectors is None:
            field_vectors = encoder.encode(raw, progress=self.progress)
        field_vectors = np.asarray(field_vectors)
        if field_vectors.shape != (len(raw), 12, 768):
            raise ValueError("Frozen original encoder vectors must be aligned n×12×768")
        self.encoder_ = encoder
        if prefitted_base is None:
            self.base_ = OriginalFallbackHybrid(specifications=self.specifications,
                outer_fold=self.outer_fold, n_jobs=self.n_jobs,
                polybot_factory=OriginalPolybotClean, progress=self.progress)
            self.base_.fit(clean, y)
        else:
            # The private driver verifies exact canonical TRAIN inputs, labels,
            # version and specifications before supplying this fitted state.
            if not isinstance(prefitted_base, OriginalFallbackHybrid):
                raise TypeError("prefitted_base must be a validated original fitted base")
            check_is_fitted(prefitted_base, ["base_", "meta_", "fusion_", "specs_",
                                           "training_parent_values_", "n_training_examples_"])
            if (prefitted_base.outer_fold != self.outer_fold
                    or prefitted_base.n_training_examples_ != len(y)
                    or self.specifications is not None and prefitted_base.specs_ != self.specifications):
                raise ValueError("The supplied base does not match this TRAIN fold/specification")
            if np.asarray(prefitted_base.training_parent_values_).shape != (len(y), 5):
                raise ValueError("The supplied base lacks aligned inner-QUERY TRAIN parent states")
            _closure(np.asarray(prefitted_base.training_parent_values_))
            self.base_ = prefitted_base
        self._emit("original_base_ready", reused_prefitted_base=prefitted_base is not None,
                   train_n=len(y))
        current = self.base_.training_parent_values_.copy()
        del self.base_.training_parent_values_
        _closure(current)
        self.special_ = None
        if self.special_factory is not None:
            self._emit("special_physics_fit_started")
            self.special_ = self.special_factory().fit(raw, current, y, fold=self.outer_fold)
            current = _values(self.special_.predict_values(raw, current))
            _closure(current)
            self._emit("special_physics_fit_complete")
        self.filler_ = None
        if self.filler_factory is not None:
            self._emit("filler_physics_fit_started")
            self.filler_ = self.filler_factory().fit(raw, current, y)
            current = _values(self.filler_.predict_values(raw, current))
            _closure(current)
            self._emit("filler_physics_fit_complete")
        selected = np.flatnonzero(active)
        if len(selected) < 24:
            raise ValueError("Original late-head training requires adequate non-platform controls")
        splits = list(KFold(3, shuffle=True, random_state=20261012).split(selected))
        self.stages_, self.stage_diagnostics_ = [], []
        for kind in ("global_kernel", *LATE_HEADS):
            stage_start = time.monotonic()
            residual = y-current[:, 0]
            delta_oof = np.zeros(len(raw))
            self._emit("original_residual_stage_started", kind=kind)
            for inner, (fit_pos, query_pos) in enumerate(splits):
                tr, qu = selected[fit_pos], selected[query_pos]
                if kind == "global_kernel":
                    fitted = OriginalGlobalKernel(outer_fold=inner).fit(
                        _slice(science, tr), current[tr][:, PARENT4], residual[tr])
                    delta_oof[qu] = fitted.predict_delta(_slice(science, qu), current[qu][:, PARENT4])
                else:
                    fitted, _ = fit_original_head(kind, _slice(raw, tr), _slice(science, tr),
                        current[tr][:, PARENT4], residual[tr], fold=inner,
                        field_vectors_train=field_vectors[tr] if kind == "matbert" else None)
                    delta_oof[qu] = predict_delta(fitted, _slice(raw, qu), _slice(science, qu),
                        current[qu][:, PARENT4], field_vectors=field_vectors[qu] if kind == "matbert" else None)
                del fitted
            denominator = float(delta_oof[selected] @ delta_oof[selected])
            weight = float(np.clip(residual[selected] @ delta_oof[selected] / denominator,
                                   -1., 1.)) if denominator > 1e-12 else 0.
            if kind == "global_kernel":
                state = OriginalGlobalKernel(outer_fold=self.outer_fold).fit(
                    _slice(science, selected), current[selected][:, PARENT4], residual[selected])
            else:
                state, _ = fit_original_head(kind, _slice(raw, selected), _slice(science, selected),
                    current[selected][:, PARENT4], residual[selected], fold=self.outer_fold,
                    field_vectors_train=field_vectors[selected] if kind == "matbert" else None)
            self.stages_.append({"kind": kind, "state": state, "weight": weight})
            current[:, 0] += weight*delta_oof
            current[:, 3] += weight*delta_oof
            _closure(current)
            diag = {"kind": kind, "TRAIN_internal_scalar_weight": weight,
                    "n_training_examples": len(selected), "seconds": time.monotonic()-stage_start,
                    "weight_selection": "One bounded scalar learned using three inner-fold deltas inside outer TRAIN only"}
            self.stage_diagnostics_.append(diag)
            self._emit("original_residual_stage_complete", **diag)
        self._emit("original_P_fit_started")
        self.physics_, _, physics_diag = fit_original_physics(
            _slice(raw, selected), current[selected], y[selected], fold=self.outer_fold)
        self.physics_summary_ = {"status": self.physics_["status"],
            "n_eligible_TRAIN": int(np.sum(physics_diag["eligible"])),
            "specialized_unsupported_protocols": "Retained parent M, including its explicitly supported special physics"}
        self.n_training_examples_ = len(y)
        self.fit_seconds_ = time.monotonic()-started
        self.progress = None
        return self

    def predict_values(self, X, *, field_vectors=None, scientific=None, prefer_physics=None):
        check_is_fitted(self, ["base_", "stages_", "physics_"])
        clean, raw, science, active = self._project(list(X), scientific)
        if not raw:
            return np.empty((0, 5))
        values = self.base_.predict_values(clean)
        if self.special_ is not None:
            values = _values(self.special_.predict_values(raw, values))
        if self.filler_ is not None:
            values = _values(self.filler_.predict_values(raw, values))
        selected = np.flatnonzero(active)
        if len(selected):
            if field_vectors is None:
                field_vectors = self.encoder_.encode(raw)
            for stage in self.stages_:
                if not stage["weight"]:
                    continue
                if stage["kind"] == "global_kernel":
                    delta = stage["state"].predict_delta(_slice(science, selected), values[selected][:, PARENT4])
                else:
                    delta = predict_delta(stage["state"], _slice(raw, selected), _slice(science, selected),
                        values[selected][:, PARENT4],
                        field_vectors=np.asarray(field_vectors)[selected] if stage["kind"] == "matbert" else None)
                values[selected, 0] += stage["weight"]*delta
                values[selected, 3] += stage["weight"]*delta
            choose_P = self.prefer_physics if prefer_physics is None else prefer_physics
            if choose_P:
                values[selected], _ = predict_original_physics(self.physics_,
                    _slice(raw, selected), values[selected])
        _closure(values)
        return values

    def predict(self, X):
        return self.predict_values(X)[:, 0]

    def predict_conductivity(self, X):
        return np.power(10., self.predict(X))

    def summary(self):
        check_is_fitted(self, ["stages_", "physics_"])
        return {"version": VERSION, "n_training_examples": self.n_training_examples_,
            "input_version": self.input_version_,
            "fit_seconds": self.fit_seconds_, "base": self.base_.summary(),
            "special_physics": self.special_.summary() if self.special_ is not None else {"status": "not_connected"},
            "filler_and_DAE_physics": self.filler_.summary() if self.filler_ is not None else {"status": "not_connected"},
            "late_heads": self.stage_diagnostics_, "physical_P": self.physics_summary_,
            "source_features": False, "same_assay_SD_features": False,
            "observed_coverage_features": False, "query_target_access": False,
            "complete_old_weights_unchanged": False,
            "scope": "Original source-free/default algorithms plus retained input-supported physics; refitted within TRAIN; unsupported source/observed branches removed or fall back.",
            "prediction_units": "log10(S/cm); predict_conductivity converts to S/cm"}
