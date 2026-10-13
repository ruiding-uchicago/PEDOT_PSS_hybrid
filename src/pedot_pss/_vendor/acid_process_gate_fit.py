"""Training-fold-only calibration of a shared effective acid process gate.

No source/row IDs, query targets, files, model loading or base refitting. The
supplied DomainTransport is frozen; only eta/K/beta are estimated from explicit
training log10(S/cm). A null eta=0 candidate competes on the SAME penalized
objective. This calibration does not identify PSS loss or microscopic kinetics.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.optimize import least_squares

try:
    from . import acid_process_gate as gate
except ImportError:
    from . import acid_process_gate as gate
VERSION = "acid-process-gate-fold-calibration-v1"
NULL_PARAMETERS = dict(eta=0., half_saturation_mol_L=1., hydration_exponent=1.)
OBJECTIVE_TIE_RELATIVE_TOLERANCE = 1e-12
FIXED_STARTS = ((.5, 1., 1.), (.25, .5, .5), (.75, 5., 2.))


def _vector(value, n, name):
    raw = np.asarray(value)
    if raw.shape != (n,) or raw.dtype.kind not in "iuf":
        raise ValueError(name + " must be an aligned real numeric vector")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _squares(value):
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(value @ value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite squared-error objective")
    return result


def _base_binding(base):
    """Small ordinary parameter binding, not a file/content authentication."""
    gem = base.gem_
    return dict(mode=base.mode, interface_scale=float(base.interface_scale),
        activation_energy_eV=float(base.activation_energy_eV),
        gem_coefficients=np.asarray(gem.coef_, float).tolist(),
        domain_coefficients=np.asarray(base.domain_parameters_, float).tolist(),
        gem_fixed=dict(fraction=gem.fraction, threshold=gem.threshold, exponent=gem.exponent,
                       sigma_high=gem.sigma_high, low_log10_min=gem.low_log10_min,
                       enable_hopping=gem.enable_hopping))


def _magnitude(value):
    absolute = abs(value)
    return dict(n_exact_nonzero=int(np.count_nonzero(value)),
                n_above_tolerance=int(np.count_nonzero(absolute > gate.CHECK_TOLERANCE)),
                maximum_abs=float(np.max(absolute)) if len(value) else 0.)


def _response(parts, baseline, admitted):
    delta = parts["prediction_log10_S_cm"]-baseline["prediction_log10_S_cm"]
    gem_delta = parts["gem_log10"]-baseline["gem_log10"]
    rho_delta = parts["acid_process_new_log_rho_linear"]-parts["acid_process_old_log_rho_linear"]
    return dict(conductivity_log10=_magnitude(delta), gem_log10=_magnitude(gem_delta),
        linear_log_rho=_magnitude(rho_delta),
        n_increased_conductivity=int(np.count_nonzero(delta > gate.CHECK_TOLERANCE)),
        n_decreased_conductivity=int(np.count_nonzero(delta < -gate.CHECK_TOLERANCE)),
        n_changed_outside_admission=int(np.count_nonzero(delta[~admitted])),
        n_changed_coordinates=int(parts["acid_process_coordinate_changed"].sum()),
        circuit_residual_max_abs=gate._maximum_abs(parts["closure_residual"]),
        gem_equation_residual_max_abs=gate._maximum_abs(parts["gem_equation_residual"]),
        n_phase_bound_violations=int(parts["gem_phase_bound_violation"].sum()),
        n_domain_bound_violations=int(parts["domain_bound_violation"].sum()))


class AcidProcessGateCalibrator:
    """A three-parameter fold-local fit; the frozen base is never stored/refit.

    The objective uses admitted coordinate-responsive positive-weight TRAINING
    rows only; all other rows have an invariant physical prediction. Weights
    are normalized within this objective subset. Penalty is
    n_eligible * regularization * eta**2, with NO artificial K/beta null prior.
    If candidates tie within the declared floating objective tolerance, select
    the exact null. That selection is distinct from thresholding forward values.
    """
    def __init__(self, regularization=.02, max_nfev=120, protocol_scope=gate.PROTOCOL_SCOPE):
        if not gate.acid._finite_number(regularization) or regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if type(max_nfev) is not int or max_nfev < 1:
            raise ValueError("max_nfev must be a positive integer")
        if protocol_scope != gate.PROTOCOL_SCOPE:
            raise ValueError("Explicit verified acid protocol scope required")
        self.regularization, self.max_nfev = float(regularization), max_nfev
        self.protocol_scope = protocol_scope

    def fit(self, physical_rows, acid_rows, y, *, base, protocol_admitted, sample_weight=None):
        physical_rows, acid_rows = list(physical_rows), list(acid_rows)
        n = len(physical_rows)
        if not n:
            raise ValueError("Nonempty explicit training rows required")
        target = _vector(y, n, "Training log10(S/cm) y")
        weights = np.ones(n) if sample_weight is None else _vector(sample_weight, n, "Training sample weights")
        with np.errstate(over="ignore", invalid="ignore"):
            total_weight = float(weights.sum())
        if (weights < 0).any() or not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Training sample weights must have finite positive total and be nonnegative")
        baseline = gate.apply_acid_process_gate(physical_rows, acid_rows, base, NULL_PARAMETERS,
            protocol_admitted=protocol_admitted, protocol_scope=self.protocol_scope)
        binding = _base_binding(base)
        # Validate all explicit training targets before optimizing any subset.
        baseline_sse = _squares(baseline["prediction_log10_S_cm"]-target)
        probe = gate.apply_acid_process_gate(physical_rows, acid_rows, base,
            dict(NULL_PARAMETERS, eta=1.), protocol_admitted=protocol_admitted, protocol_scope=self.protocol_scope)
        admitted = baseline["acid_process_is_admitted"] == 1.
        eligible = admitted & (probe["acid_process_coordinate_changed"] == 1.) & (weights > 0)
        indices = np.flatnonzero(eligible)
        parameters = dict(NULL_PARAMETERS)
        optimization = dict(ran=False, success=True, evaluations=0, status="no_eligible_training_rows",
            message="Exact null; no admitted coordinate-responsive positive-weight training rows.",
            selected_candidate="null", null_objective=0., optimized_objective=None, selected_objective=0.,
            objective_improvement_over_null=0., objective_tie_tolerance=OBJECTIVE_TIE_RELATIVE_TOLERANCE,
            starts=[], all_start_optimizations_successful=True)
        if len(indices):
            selected_physical = [physical_rows[i] for i in indices]
            selected_acids = [acid_rows[i] for i in indices]
            protocol = np.ones(len(indices), dtype=bool)
            root_weight = np.sqrt(weights[indices]/weights[indices].mean())
            penalty_scale = math.sqrt(len(indices)*self.regularization)

            def unpack(vector):
                if np.asarray(vector).shape != (3,) or not np.isfinite(vector).all():
                    raise ValueError("Nonfinite optimizer parameter vector")
                return dict(eta=float(vector[0]),
                    half_saturation_mol_L=min(100., max(.01, math.exp(float(vector[1])))),
                    hydration_exponent=min(8., max(.1, math.exp(float(vector[2])))))

            def objective_parameters(pars):
                parts = gate.apply_acid_process_gate(selected_physical, selected_acids, base, pars,
                    protocol_admitted=protocol, protocol_scope=self.protocol_scope)
                residual = root_weight*(parts["prediction_log10_S_cm"]-target[indices])
                result = np.r_[residual, penalty_scale*pars["eta"]]
                _squares(result)
                return result

            null_objective = _squares(objective_parameters(NULL_PARAMETERS))
            starts, candidates = [], []
            for eta, half, exponent in FIXED_STARTS:
                solution = least_squares(lambda vector: objective_parameters(unpack(vector)),
                    [eta, math.log(half), math.log(exponent)],
                    bounds=([0., math.log(.01), math.log(.1)], [1., math.log(100.), math.log(8.)]),
                    max_nfev=self.max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                candidate = unpack(solution.x)
                value = _squares(objective_parameters(candidate))
                starts.append(dict(initial=dict(eta=eta, half_saturation_mol_L=half, hydration_exponent=exponent),
                    parameters=candidate, objective=value, success=bool(solution.success), evaluations=int(solution.nfev),
                    status=int(solution.status), message=str(solution.message)))
                candidates.append((value, candidate, solution))
            optimized_objective, candidate, solution = min(candidates, key=lambda item: item[0])
            tie = OBJECTIVE_TIE_RELATIVE_TOLERANCE*max(1., null_objective)
            select_optimized = optimized_objective < null_objective-tie
            if select_optimized:
                parameters = candidate
            selected_objective = optimized_objective if select_optimized else null_objective
            optimization = dict(ran=True, success=bool(solution.success), evaluations=sum(r["evaluations"] for r in starts),
                status=int(solution.status), message=str(solution.message),
                selected_candidate="optimized" if select_optimized else "null",
                null_objective=null_objective, optimized_objective=optimized_objective,
                selected_objective=selected_objective,
                objective_improvement_over_null=null_objective-selected_objective, objective_tie_tolerance=tie,
                starts=starts, all_start_optimizations_successful=all(r["success"] for r in starts))
        parts = gate.apply_acid_process_gate(physical_rows, acid_rows, base, parameters,
            protocol_admitted=protocol_admitted, protocol_scope=self.protocol_scope)
        if _base_binding(base) != binding:
            raise ValueError("Frozen base parameters changed during calibration")
        selected_sse = _squares(parts["prediction_log10_S_cm"]-target)
        old_linear = baseline["acid_process_old_log_rho_linear"]
        probe_response = _response(probe, baseline, admitted)
        selected_response = _response(parts, baseline, admitted)
        details = dict(version=VERSION, forward_version=gate.VERSION, input_version=gate.INPUT_VERSION,
            protocol_scope=self.protocol_scope, parameters=parameters, bounds=deepcopy(gate.PARAMETER_BOUNDS),
            regularization=self.regularization, max_nfev=self.max_nfev,
            regularization_center=dict(eta=0., log_K=None, log_beta=None),
            objective_scope="Admitted coordinate-responsive positive-weight training rows; normalized weights; penalty=n_eligible*regularization*eta^2.",
            n_training_examples=n, n_supported_training_examples=int((baseline["acid_process_has_supported_concentration"] == 1.).sum()),
            n_protocol_admitted_training_examples=int((baseline["acid_process_protocol_admitted"] == 1.).sum()),
            n_admitted_training_examples=int(admitted.sum()), n_eligible_training_examples=len(indices),
            n_effective_training_examples=int(np.count_nonzero(eligible &
                (abs(probe["acid_process_correction_log10"]) > gate.CHECK_TOLERANCE))),
            n_zero_weight_training_examples=int(np.count_nonzero(weights == 0)), optimization=optimization,
            frozen_base_parameters=binding,
            frozen_acid_coefficients=dict(gem_process_score=float(base.gem_.coef_[gate.GEM_ACID_COLUMN]),
                domain_barrier_relief=float(base.domain_parameters_[gate.DOMAIN_ACID_COLUMN]),
                interpretation="Signs are recorded, not estimated or used to admit rows. Positive GEM acid coef can lower GEM when gated down; positive barrier-relief coef increases rho when gated down."),
            baseline_clip=dict(n_log_rho_at_or_outside_clip=int(np.count_nonzero(abs(old_linear) >= 30.)),
                n_admitted_at_or_outside_clip=int(np.count_nonzero(admitted & (abs(old_linear) >= 30.))),
                n_admitted_eta1_at_or_outside_clip=int(np.count_nonzero(admitted &
                    (abs(probe["acid_process_new_log_rho_linear"]) >= 30.))),
                interpretation="Original circuit clip is retained by full re-solve; no exact linear cancellation is claimed."),
            null_training_rmse_log10=math.sqrt(baseline_sse/n), selected_training_rmse_log10=math.sqrt(selected_sse/n),
            eta1_probe_response=probe_response, selected_response=selected_response,
            physics_effect_detected=selected_response["conductivity_log10"]["n_above_tolerance"] > 0,
            effect_detection_tolerance=gate.CHECK_TOLERANCE,
            limits=["Only the explicitly supplied training targets are fit; caller owns fold membership and verified input protocol.",
                "Frozen base, thermal/hopping/bypass coordinates and old coefficients are never fit or stored as a live model here.",
                "Shared eta/K/beta are effective process proxies, not measured PSS loss, dissociation, water activity or validated microscopic constants.",
                "Initial neat-endpoint proxy applies only to three named monoacids; all six share saturation K.",
                "Nonconverged optimization is reported as such even when its candidate beats the null; no automatic retry.",
                "Exact nonzero response can be numerical noise; above-tolerance counts are supplied separately."])
        # Commit only small serializable state; no training rows/y/base retained.
        self.parameters_, self.base_binding_, self.details_ = dict(parameters), binding, details
        self.n_training_examples_ = n
        return self

    def components(self, physical_rows, acid_rows, *, base, protocol_admitted):
        if not hasattr(self, "parameters_"):
            raise RuntimeError("Calibrate on explicit training rows first")
        if _base_binding(base) != self.base_binding_:
            raise ValueError("Query base differs from the frozen training-fold base")
        return gate.apply_acid_process_gate(physical_rows, acid_rows, base, self.parameters_,
            protocol_admitted=protocol_admitted, protocol_scope=self.protocol_scope)

    def predict(self, physical_rows, acid_rows, *, base, protocol_admitted):
        return self.components(physical_rows, acid_rows, base=base,
                               protocol_admitted=protocol_admitted)["prediction_log10_S_cm"]

    def summary(self):
        if not hasattr(self, "details_"):
            raise RuntimeError("Calibrate on explicit training rows first")
        result = deepcopy(self.details_)
        result["parameters"] = dict(self.parameters_)
        return result
