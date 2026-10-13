"""Row-specific effective connectivity/state -> actual GEM -> frozen circuit.

No IO, fitting/loading, protocol/identity routing, output patch or model writes.
Caller supplies thresholds and must enforce TRAIN membership for inverse/loss.
Effective low phase and threshold are not measured PSS/morphological quantities.
Do not reuse old fixed-threshold warm.curvature_weights with these inverses:
TRAIN inverse/offset/curvature/objective and query must use matching thresholds.
"""
from collections.abc import Mapping
import math

import numpy as np
from scipy.special import expit

try:
    from . import phase_state_forward as old
    from . import solvent_connectivity_boundary as connectivity
except ImportError:
    from . import phase_state_forward as old
    from . import solvent_connectivity_boundary as connectivity
network = old.network
VERSION = "row-connectivity-effective-phase-actual-forward-v1"
MIN_LOG_LOW_S_CM = old.MIN_LOG_LOW_S_CM
CHECK_TOLERANCE = old.CHECK_TOLERANCE
INVERSE_INTERVAL_TOLERANCE = old.INVERSE_INTERVAL_TOLERANCE
LN10 = math.log(10.)


def _vector(values, n, name):
    # Hot path: numeric ndarrays cannot contain Boolean-typed elements.
    # Earlier upstream float coercion loses provenance; it cannot be recovered.
    if isinstance(values, np.ndarray) and values.dtype.kind in "iuf":
        return network._array(values, n, name)
    # Reject mixed [True, .2] BEFORE coercion; inspect list/object inputs only.
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(values, dtype=object).flat):
        raise ValueError("Non-Boolean numeric vector required: " + name)
    return network._array(values, n, name)


def phase_configuration(gem_model):
    """Read frozen geometry; f may equal/exceed the supplied per-row fc."""
    high, fraction, threshold, exponent = connectivity.geometry(gem_model)
    if high <= 10. ** MIN_LOG_LOW_S_CM: raise ValueError("Positive phase-state span required")
    return dict(sigma_high_S_cm=high, fraction=fraction, threshold=threshold, exponent=exponent,
        min_log_low_S_cm=MIN_LOG_LOW_S_CM, max_log_low_S_cm=math.log10(high))


def _inputs(values, reference_parts, gem_model, thresholds, name):
    raw = np.asarray(values)
    if raw.ndim != 1: raise ValueError("Aligned one-dimensional " + name + " required")
    values = _vector(values, len(raw), name)
    threshold = _vector(thresholds, len(raw), "thresholds")
    if ((threshold <= 0.) | (threshold >= 1.)).any(): raise ValueError("Strict 0<fc<1 required")
    if not isinstance(reference_parts, Mapping): raise ValueError("Actual reference mapping required")
    ref = old._context({k:_vector(v, len(raw), k) for k,v in reference_parts.items()}, len(raw))
    return values, ref, phase_configuration(gem_model), threshold


def _forward(state, ref, config, thresholds):
    low_bound, high_bound = config["min_log_low_S_cm"], config["max_log_low_S_cm"]
    if ((state < low_bound) | (state > high_bound)).any(): raise ValueError("Low-phase state outside bounds; no implicit clipping")
    low = 10. ** state
    low[state == high_bound] = config["sigma_high_S_cm"]
    physics = network.transport._physics
    g = physics.gem_conductivity(low, config["sigma_high_S_cm"], config["fraction"], thresholds, config["exponent"])
    out = network.network_with_rho(np.log10(g), ref["series_resistivity_ohm_cm"], ref["bypass_fraction"], anisotropy_factor=ref.get("anisotropy_factor"))
    actual = out["sigma_gem_S_cm"]
    residual = physics.gem_residual(actual, low, config["sigma_high_S_cm"], config["fraction"], thresholds, config["exponent"])
    violation = ((actual < low * (1. - 1e-11)) | (actual > config["sigma_high_S_cm"] * (1. + 1e-11))).astype(float)
    if not np.isfinite(residual).all() or (abs(residual) > CHECK_TOLERANCE).any() or violation.any(): raise ValueError("Actual GEM equation/phase bounds failed")
    out.update(gem_equation_residual=residual, gem_phase_bound_violation=violation,
        effective_low_phase_S_cm=low, phase_state_log_low_S_cm=state.copy(), phase_connectivity_threshold=thresholds.copy())
    return out


def _traces(out, ref):
    out.update({"phase_connectivity_reference_" + k:v.copy() for k,v in ref.items()})
    for key, label in (("prediction_log10_S_cm", "correction"), ("gem_log10", "gem_correction")):
        if key in ref: out["phase_connectivity_" + label + "_from_reference_log10"] = out[key] - ref[key]
    return out


def solve_phase_state(log_low, reference_parts, gem_model, *, thresholds):
    """Query forward, no target argument. State is log10(S/cm); fc dimensionless.

    Effective rho[ohm cm]/p are frozen, already scaled/clipped/anisotropic.
    They are not rescaled or clipped again. Actual geometry/coef stay unchanged.
    All old P4/gate trace fields appear only as phase_connectivity_reference_*;
    new GEM/circuit values are actual solves, not relabelled old gate effects.
    """
    state, ref, config, fc = _inputs(log_low, reference_parts, gem_model, thresholds, "low-phase state")
    return _traces(_forward(state, ref, config, fc), ref)


def forward_with_derivative(raw_scores, reference_parts, gem_model, *, thresholds, offset=0.):
    """Bounded sigmoid state and actual dlog10(sigma)/draw at fixed row fc.

    The caller-frozen TRAIN offset is one scalar, also used at query. With
    A_i=(1-fc_i)/fc_i, implicit-GEM slope U/(U+V) uses
    U=(1-f)l/(l+A_i)^2, V=f*h/(h+A_i)^2. The positive circuit and sigmoid
    slopes multiply it; ln10 factors cancel. This is not d(sigma)/d(fc).
    """
    raw, ref, config, fc = _inputs(raw_scores, reference_parts, gem_model, thresholds, "raw scores")
    if not network._finite(offset): raise ValueError("One finite non-Boolean TRAIN offset required")
    offset = float(offset)
    with np.errstate(over="ignore"): shifted = raw + offset
    if not np.isfinite(shifted).all(): raise ValueError("Finite raw+offset required")
    probability = expit(shifted); lo, hi = config["min_log_low_S_cm"], config["max_log_low_S_cm"]
    state = np.clip(lo + (hi - lo) * probability, lo, hi)
    out = _traces(_forward(state, ref, config, fc), ref)
    f, t = config["fraction"], config["exponent"]
    log_a = np.log((1. - fc) / fc)
    log_l, log_h = (state - out["gem_log10"]) * LN10 / t, (hi - out["gem_log10"]) * LN10 / t
    log_u = (-math.inf if f == 1. else math.log1p(-f)) + log_l - 2. * np.logaddexp(log_l, log_a)
    log_v = (-math.inf if f == 0. else math.log(f)) + log_h - 2. * np.logaddexp(log_h, log_a)
    gem_slope = np.exp(log_u - np.logaddexp(log_u, log_v))
    rho, p = out["series_resistivity_ohm_cm"], out["bypass_fraction"]
    with np.errstate(divide="ignore"):
        inverse_load = -np.logaddexp(0., np.log(rho) + out["gem_log10"] * LN10)
        log_p, log_not_p = np.log(p), np.log1p(-p)
    circuit_slope = np.exp(np.logaddexp(log_p, log_not_p + 2. * inverse_load) - np.logaddexp(log_p, log_not_p + inverse_load))
    state_slope = (hi - lo) * probability * expit(-shifted)
    derivative = gem_slope * circuit_slope * state_slope
    if (any(not np.isfinite(v).all() for v in (gem_slope, circuit_slope, state_slope, derivative))
            or ((gem_slope < 0.) | (gem_slope > 1. + 1e-12) | (circuit_slope < 0.) | (circuit_slope > 1. + 1e-12) | (derivative < 0.)).any()): raise ValueError("Finite nonnegative physical-chain derivatives required")
    out.update(phase_raw_score=raw.copy(), phase_raw_offset=np.full(len(raw), offset), phase_sigmoid_probability=probability,
        dlog_gem_dlog_low=gem_slope, dlog_sigma_dlog_gem=circuit_slope, dlog_sigma_dlog_low=gem_slope * circuit_slope,
        dlog_low_draw=state_slope, dprediction_draw=derivative)
    return out


def train_target_inverse(y_train, reference_parts, gem_model, *, thresholds, iterations=60):
    """TRAIN-only projected latent supervision, not measured phase conductivity.

    The caller enforces fold membership; this pure function cannot certify
    labels came from training. NEVER invert heldout labels. Both endpoints,
    every bisection and final replay use this SAME row-aligned threshold vector.
    Returns the original 11-field schema; signed error is projected-original.
    """
    y, ref, config, fc = _inputs(y_train, reference_parts, gem_model, thresholds, "training targets")
    if type(iterations) is not int or not 40 <= iterations <= 128: raise ValueError("40..128 integer bisection iterations required")
    lower, upper = np.full(len(y), config["min_log_low_S_cm"]), np.full(len(y), config["max_log_low_S_cm"])
    minimum, maximum = (_forward(v, ref, config, fc)["prediction_log10_S_cm"] for v in (lower, upper))
    if (maximum - minimum <= INVERSE_INTERVAL_TOLERANCE).any(): raise ValueError("Numerically distinguishable monotone interval required")
    below, above = y < minimum, y > maximum; projected = np.clip(y, minimum, maximum)
    lo, hi = lower.copy(), upper.copy()
    for _ in range(iterations):
        mid = lo + (hi - lo) / 2.; left = _forward(mid, ref, config, fc)["prediction_log10_S_cm"] < projected
        lo[left] = mid[left]; hi[~left] = mid[~left]
    state = lo + (hi - lo) / 2.
    state[projected == minimum] = lower[projected == minimum]; state[projected == maximum] = upper[projected == maximum]
    replay = _forward(state, ref, config, fc)["prediction_log10_S_cm"]
    if (abs(replay - projected) > 1e-9).any(): raise ValueError("Actual inverse replay exceeds1e-9")
    return dict(latent_log_low_S_cm=state, projected_target_log10_S_cm=projected,
        target_below_attainable_range=below, target_above_attainable_range=above, target_was_projected=below | above,
        target_within_attainable_range=~(below | above), projection_error_log10=projected - y,
        attainable_lower_log10_S_cm=minimum, attainable_upper_log10_S_cm=maximum, reconstructed_log10_S_cm=replay,
        inverse_replay_error_log10=replay - projected)


def training_squared_objective(raw_scores, y_train, reference_parts, gem_model, *, thresholds, offset=0., hessian_floor=1e-6, sample_weight=None):
    """TRAIN-only squared-response gradient and positive GN (not exact) Hessian.

    Same explicit thresholds throughout; g=2w(P-y)P', H=w*max(2P'^2,floor).
    A half-square caller must multiply BOTH by.5; floor precedes sample weights.
    Only caller can enforce TRAIN membership. Query uses forward without y.
    """
    out = forward_with_derivative(raw_scores, reference_parts, gem_model, thresholds=thresholds, offset=offset)
    n = len(out["prediction_log10_S_cm"]); y = _vector(y_train, n, "TRAIN labels")
    if not network._finite(hessian_floor) or hessian_floor <= 0.: raise ValueError("Positive finite non-Boolean Hessian floor required")
    weights = np.ones(n) if sample_weight is None else _vector(sample_weight, n, "sample weights")
    if (weights < 0.).any(): raise ValueError("Nonnegative sample weights required")
    derivative = out["dprediction_draw"]
    with np.errstate(over="ignore", invalid="ignore"):
        gradient = 2. * weights * (out["prediction_log10_S_cm"] - y) * derivative
        hessian = weights * np.maximum(2. * derivative**2, float(hessian_floor))
    if not np.isfinite(gradient).all() or not np.isfinite(hessian).all(): raise ValueError("Finite loss gradient/GN Hessian required")
    return gradient, hessian
