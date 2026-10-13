"""Retained original SWCNT positive parallel path and explicit pure-DAE prior.

Admission uses only projected experiment fields and the original closed
protocol/basis/UV parsers. Missing pressure, mass basis, forming protocol or
illumination does not activate a branch. No source, row ID, old OOF or corpus
lookup is performed by this model. fit accepts only caller-supplied TRAIN data.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


from .clean_inputs import canonicalize, original_view, feature_view
from .filler_parallel_inputs import filler_parallel_input, GRADE, CNT
from .filler_parallel_forward import forward, PARAMETER_BOUNDS
from .pure_dae_insulator_prior import pure_dae_admission, compose, NOMINAL_LOG10_SIGMA_S_CM
from .original_polybot_clean import eligible_scientific

# Original filler_parallel_cv physical arm P, without its historical-file runner.
P_NAMES = ("b", "A", "kappa0", "kappa1")
P_LOWER = np.array([0., 0., np.log(.5), -np.log(100.)])
P_UPPER = np.array([12., np.log1p(1e6), np.log(200.), np.log(100.)])
P_SCALES = P_UPPER.copy()
P_STARTS = ([1., 100., np.log(10.), 0.],
            [4., 1000., np.log(50.), np.log(3.)],
            [0., 10., np.log(2.), -np.log(3.)])
REGULARIZATION, MAX_NFEV, MIN_TRAIN = .003, 200, 16
_PURE_DAE_PROCESS = re.compile(r"(?:^|;\s*)pure DAE film(?: control)?(?:\s*;|\s*$)", re.I)


def _admission_view(raw):
    """Restore material-bearing geometry from explicit clean condition facts.

    The public contract normalizes geometry to 'film'. These two adaptations
    re-express material facts already present in grade/additives/process for
    the original literal closed-scope parsers; they infer no amounts/protocols.
    """
    canonical = canonicalize(raw)
    view = original_view(canonical)
    if (canonical["specimen_geometry"] == "film"
            and GRADE.fullmatch(canonical["polymer_grade"] or "")
            and any(CNT.fullmatch(name) for name in canonical["additive_names"])):
        view["specimen_geometry"] = "PEDOT:PSS/SWCNT composite thin film"
    if (canonical["specimen_geometry"] == "film"
            and _PURE_DAE_PROCESS.search(canonical["process_sequence"] or "")):
        view["specimen_geometry"] = "pure DAE film"
    return canonical, view


def input_admission(rawrows):
    """Return target-free masks/nominal coordinates and aggregate reasons."""
    rawrows = list(rawrows)
    filler, dae, pb = [], [], []
    for raw in rawrows:
        canonical, view = _admission_view(raw)
        filler_rule = filler_parallel_input(view)
        dae_rule = pure_dae_admission(view)
        filler.append(filler_rule)
        dae.append(dae_rule)
        # Only candidate admissions need a PB-overlap veto. All outside rows
        # are exact copies independently of their platform membership.
        pb.append(eligible_scientific(feature_view(canonical))
                  if filler_rule["active"] or dae_rule["active"] else False)
    mask = np.array([row["active"] for row in filler], dtype=bool)
    dae_mask = np.array([row["active"] for row in dae], dtype=bool)
    pb_mask = np.array(pb, dtype=bool)
    if np.any((mask | dae_mask) & pb_mask) or np.any(mask & dae_mask):
        raise ValueError("Closed filler/DAE/PB experiment scopes must not overlap")
    return {
        "x": np.array([row["x_nominal_mass_fraction"] if row["active"] else np.nan
                        for row in filler], dtype=float),
        "z": np.array([row["z_filtration_coldpress"] == 1 if row["active"] else False
                        for row in filler], dtype=bool),
        "admission_mask": mask, "dae_mask": dae_mask, "polybot_mask": pb_mask,
        "counts": {"n_input": len(rawrows), "n_filler": int(mask.sum()),
                   "n_dae_uv_on": int(dae_mask.sum()),
                   "n_vacuum_filter_40MPa_10min": int(sum(row["active"] and row["z_filtration_coldpress"] == 1 for row in filler)),
                   "n_direct_spin": int(sum(row["active"] and row["z_filtration_coldpress"] == 0 for row in filler)),
                   "filler_reasons": dict(Counter(row["reason"] for row in filler)),
                   "dae_reasons": dict(Counter(row["reason"] for row in dae))},
    }


def _parent(values, n):
    values = np.asarray(values)
    if (values.shape != (n, 5) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()):
        raise ValueError("Finite aligned current five-field parent values required")
    values = np.asarray(values, dtype=float)
    if np.any(abs(values[:, 0] - values[:, 2] - values[:, 3]) > 1e-10):
        raise ValueError("Current parent decomposition must close")
    return values


def _parameters(optimizer):
    parameters = np.asarray(optimizer, dtype=float).copy()
    parameters[1] = np.expm1(parameters[1])
    if (parameters.shape != (4,) or not np.isfinite(parameters).all()
            or any(not PARAMETER_BOUNDS[name][0] <= value <= PARAMETER_BOUNDS[name][1]
                   for name, value in zip(P_NAMES, parameters))):
        raise ValueError("Original filler parameter bounds exceeded")
    return parameters


def _physical(host, coordinates, parameters):
    return forward(host, coordinates["x"], coordinates["z"], coordinates["admission_mask"],
                   **dict(zip(P_NAMES, parameters)))


def _fit_original_physical(base, y, host, coordinates):
    """Exact original physical fit objective, starts, null and bounds."""
    n = len(y)
    def objective(optimizer):
        delta = _physical(host, coordinates, _parameters(optimizer))["total_correction_log10"]
        return np.r_[base + delta - y, np.sqrt(n * REGULARIZATION) * optimizer / P_SCALES]
    selected = np.zeros(4)
    initial = objective(selected)
    cost, winner = float(initial @ initial), 0
    candidates = [{"kind": "exact_null", "success": True, "nfev": 0, "objective": cost}]
    if n >= MIN_TRAIN:
        for start in P_STARTS:
            optimizer = np.asarray(start, dtype=float).copy()
            optimizer[1] = np.log1p(optimizer[1])
            fit = least_squares(objective, optimizer, bounds=(P_LOWER, P_UPPER),
                                max_nfev=MAX_NFEV, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            value = float(fit.fun @ fit.fun)
            candidates.append({"kind": "bounded_LS", "success": bool(fit.success),
                               "status": int(fit.status), "nfev": int(fit.nfev), "objective": value})
            if fit.success and np.isfinite(value) and value < cost:
                selected, cost, winner = fit.x.copy(), value, len(candidates) - 1
    return _parameters(selected), {"n_train": n, "selected_start": winner,
        "optimizer_coordinates": selected.tolist(), "candidates": candidates,
        "min_train": MIN_TRAIN, "max_nfev": MAX_NFEV,
        "regularization": REGULARIZATION, "microscopic_identification_claim": False}


class OriginalFillerAndDAE(BaseEstimator):
    """Original narrow physical branches composed with a current clean parent.

    The filler difference is fitted to current TRAIN parent errors and applied
    to its actual physical host. It retains the old nominal-mass hypothesis,
    not an asserted CNT volume fraction or intrinsic CNT conductivity. DAE uses
    the original fixed -12 prior; it is not a measured/calibrated endmember.
    """
    def fit(self, raw, parentM, y):
        raw = list(raw)
        parent = _parent(parentM, len(raw))
        y = np.asarray(y)
        if y.shape != (len(raw),) or y.dtype.kind not in "iuf" or not np.isfinite(y).all():
            raise ValueError("Finite aligned TRAIN labels required")
        admitted = input_admission(raw)
        mask = admitted["admission_mask"]
        if np.any(abs(parent[mask, 1] - parent[mask, 2]) > 1e-10):
            raise ValueError("Retained filler physical increment requires parent alpha1 in its scope")
        self.parameters_ = np.zeros(4)
        self.fit_trace_ = {"n_train": 0, "selected_start": 0, "reason": "no_admitted_TRAIN_rows"}
        if mask.any():
            coords = {key: admitted[key][mask] for key in ("x", "z", "admission_mask")}
            self.parameters_, self.fit_trace_ = _fit_original_physical(
                parent[mask, 0], np.asarray(y[mask], dtype=float), parent[mask, 1], coords)
        self.training_admission_counts_ = admitted["counts"]
        return self

    def predict_values(self, raw, parentM):
        check_is_fitted(self, ("parameters_", "fit_trace_"))
        raw = list(raw)
        parent = _parent(parentM, len(raw))
        if not raw:
            return parent.copy()
        admitted = input_admission(raw)
        mask = admitted["admission_mask"]
        result = parent.copy()
        if mask.any():
            if np.any(abs(parent[mask, 1] - parent[mask, 2]) > 1e-10):
                raise ValueError("Retained filler physical increment requires parent alpha1 in its scope")
            physical = _physical(parent[:, 1], admitted, self.parameters_)
            delta = physical["total_correction_log10"]
            result[mask, 0] += delta[mask]
            result[mask, 1] += delta[mask]
            result[mask, 2] += delta[mask]
            result[mask, 4] += delta[mask]
        result = compose(result, admitted["dae_mask"], admitted["polybot_mask"],
                         log10_sigma_prior=NOMINAL_LOG10_SIGMA_S_CM)
        if (not np.isfinite(result).all()
                or np.any(abs(result[:, 0] - result[:, 2] - result[:, 3]) > 1e-10)):
            raise ValueError("Composed filler/DAE prediction must close")
        outside = ~(mask | admitted["dae_mask"])
        if not np.array_equal(result[outside], parent[outside]):
            raise ValueError("Unknown/outside physical branch must copy current parent exactly")
        return result

    def summary(self):
        check_is_fitted(self, ("parameters_", "fit_trace_"))
        return {"version": "original-filler-parallel-and-dae-clean-v1",
                "training_admission": self.training_admission_counts_,
                "physical_parameters": dict(zip(P_NAMES, self.parameters_.tolist())),
                "fit": self.fit_trace_, "dae_fixed_log10_prior": NOMINAL_LOG10_SIGMA_S_CM,
                "dae_is_measured_endmember": False, "nominal_mass_is_volume_fraction": False,
                "source_or_outcome_routing": False,
                "unknown_scope_copies_parent_exactly": True}
