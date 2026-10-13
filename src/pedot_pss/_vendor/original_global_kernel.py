"""The original joint residual model's source-independent global branch.

The original global anchors/basis and Ridge(.01) objective are retained. Once
all paper/series local kernels are removed, W=I in the original joint normal
equation; its solution is (Phi.T Phi + .01 I)^-1 Phi.T residual. There is no
population-sized zero kernel, local identity lookup, query label, or old OOF IO.

Only explicitly supplied TRAIN residuals enter fit. Parent states are outputs
of the current fitted upstream model, never observed responses or old tables.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


from .joint_kernel_features import fit_global, transform_global

GLOBAL_RIDGE = .01
REMOVED_LOCAL_RIDGE = .05  # Original parameter, traced only; no local model exists.
PARENT_COLUMNS = ("prediction_log10", "weighted_physics_log10", "residual_log10",
                  "weighted_domain_correction_log10")
_ILLEGAL_PREFIXES = ("source_", "paper_", "doi", "row_id", "sample_", "specimen_",
                     "series_", "batch_id", "target", "assay_", "observed_",
                     "context_", "structure_", "recon_source_", "recon_calibration_",
                     "cv_source_", "cv_label_", "cv_observed_", "pb_historical_",
                     "residual_only_assay_")


def _parent_matrix(parent, n):
    raw = np.asarray(parent)
    if raw.shape != (n, 4) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError("Finite aligned current-parent states with four columns required")
    return np.asarray(raw, dtype=float)


def _input_rows(scientific, parent):
    """Accept only scientific scalar features and current fitted parent states.

    Missing/nonfinite descriptors have no encoded entry. Boolean availability
    flags become explicit 0/1 numbers, as the old dictionary encoding did. No
    measurement outcome is fabricated by these container conversions.
    """
    if isinstance(scientific, (Mapping, str, bytes)):
        raise ValueError("A sequence of scientific dictionaries is required")
    scientific = list(scientific)
    parent = _parent_matrix(parent, len(scientific))
    result = []
    for original, state in zip(scientific, parent):
        if not isinstance(original, Mapping):
            raise ValueError("Each scientific feature row must be a dictionary")
        row = {}
        for key, value in original.items():
            if not isinstance(key, str):
                raise ValueError("Scientific feature names must be strings")
            lower = key.lower()
            if (lower.startswith(_ILLEGAL_PREFIXES) or "coverage" in lower
                    or "historical" in lower or lower in {"conductivity", "conductivity_s_cm"}
                    or lower.startswith("kernel_parent_")):
                continue
            if value is None:
                continue
            if isinstance(value, str):
                row[key] = value
            elif isinstance(value, (bool, np.bool_)):
                row[key] = float(value)
            elif isinstance(value, Real):
                value = float(value)
                if np.isfinite(value):
                    row[key] = value
            else:
                raise ValueError("Scientific features must be scalar numbers or strings")
        row.update({"kernel_parent_" + name: float(value)
                    for name, value in zip(PARENT_COLUMNS, state)})
        result.append(row)
    return result


class OriginalGlobalKernel(BaseEstimator):
    """TRAIN-fitted original global Gaussian-anchor residual correction.

    The caller seals the outer fold. TRAIN parent states should be upstream
    inner-OOF predictions when this is used to fit a later residual head.
    Prediction accepts only scientific features and current parent outputs.
    """
    def __init__(self, outer_fold=0):
        self.outer_fold = outer_fold

    def fit(self, scientific_clean, parent4_train, residual_train):
        if type(self.outer_fold) is not int or self.outer_fold < 0:
            raise ValueError("A nonnegative integer outer_fold is required")
        rows = _input_rows(scientific_clean, parent4_train)
        residual = np.asarray(residual_train)
        if (not rows or residual.shape != (len(rows),) or residual.dtype.kind not in "iuf"
                or not np.isfinite(residual).all()):
            raise ValueError("Finite nonempty TRAIN residuals aligned to features required")
        self.feature_model_, phi = fit_global(rows, max_centers=256, width=1.,
                                              seed=20261009 + self.outer_fold)
        residual = np.asarray(residual, dtype=float)
        lhs = phi.T @ phi + GLOBAL_RIDGE * np.eye(phi.shape[1])
        rhs = phi.T @ residual
        # Same scaled Cholesky equation used by joint_block_kernel._solve.
        scale = float(np.max(np.abs(lhs)))
        self.beta_ = cho_solve(cho_factor(lhs / scale, lower=True), rhs / scale)
        fitted = phi @ self.beta_
        gradient = phi.T @ (residual - fitted) - GLOBAL_RIDGE * self.beta_
        relative = float(np.max(np.abs(gradient), initial=0.)) / max(
            1., float(np.max(np.abs(rhs), initial=0.)),
            float(np.max(np.abs(GLOBAL_RIDGE * self.beta_), initial=0.)))
        if not np.isfinite(self.beta_).all() or relative > 1e-7:
            raise ValueError("Original global normal equation did not close")
        self.diagnostics_ = {
            "version": "original-joint-kernel-global-only-clean-v1",
            "n_train": len(rows), "n_global_features": int(phi.shape[1]),
            "n_anchors": int(phi.shape[1] - 1), "global_ridge": GLOBAL_RIDGE,
            "removed_local_ridge": REMOVED_LOCAL_RIDGE,
            "source_local_blocks_removed": True, "hidden_intercept": False,
            "constant_column_penalized": True, "width": 1.,
            "seed": 20261009 + self.outer_fold,
            "normal_equation_relative_error": relative,
            "full_population_kernel_constructed": False,
            "parent_columns": list(PARENT_COLUMNS),
        }
        return self

    def predict_delta(self, scientific_query, parent4_query):
        check_is_fitted(self, ("feature_model_", "beta_"))
        rows = _input_rows(scientific_query, parent4_query)
        prediction = transform_global(self.feature_model_, rows) @ self.beta_
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite original global kernel correction")
        return prediction

    def summary(self):
        check_is_fitted(self, "diagnostics_")
        return dict(self.diagnostics_)
