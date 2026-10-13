"""Pure P4 parameter forward and effective state extraction, never model restore.

Only explicit physical/typed inputs and small parameter summaries are accepted.
No files, fit, joblib, fold membership, saved values or experimental query y.
The caller owns provenance/protocol admission and replay against saved P4 P/G.
Returned old P4 arrays live only in reference_parts: a later tabulated network
is not a new GEM solve and must not inherit their equation residuals.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np

from . import acid_process_gate as gate
from . import domain_transport_v2 as domain
VERSION = "correlated-domain-reference-v1"
GATE_SUMMARY_VERSION = "acid-process-gate-fold-calibration-v1"
FIXED_PRIORS = {"fraction": "effective_high_phase_fraction",
    "threshold": "percolation_threshold", "exponent": "critical_exponent",
    "sigma_high": "high_phase_S_cm", "low_log10_min": "low_phase_log10_min"}
ACID_KEYS = frozenset((gate.acid.IDENTITY_KEY, gate.acid.CONCENTRATION_KEY,
                       gate.acid.HAS_KEY, gate.acid.POSITIVE_KEY))
RECORD_KEYS = frozenset(("membership", "train_targets", "values", "row_id",
                         "paper_id", "query_y", "target_log10", "target_S_cm"))


def _number(value, name, *, nonnegative=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or (nonnegative and value < 0.)):
        raise ValueError("Finite nonboolean parameter required: " + name)
    return float(value)


def _vector(value, names, name):
    if isinstance(value, Mapping):
        if set(value) != set(names):
            raise ValueError("Complete exact named coefficient schema required: " + name)
        value = [value[k] for k in names]
    raw = np.asarray(value, dtype=object)
    if raw.shape != (len(names),):
        raise ValueError("Complete coefficient vector required: " + name)
    return np.asarray([_number(x, name) for x in raw], dtype=float)


def _base(physics_summary, gate_summary):
    ps, gs = physics_summary, gate_summary
    if any(not isinstance(s, Mapping) or RECORD_KEYS.intersection(s) for s in (ps, gs)):
        raise ValueError("Plain parameter summaries required, not fold records/targets")
    if (gs["version"] != GATE_SUMMARY_VERSION or gs["forward_version"] != gate.VERSION
            or gs["input_version"] != gate.INPUT_VERSION or gs["protocol_scope"] != gate.PROTOCOL_SCOPE):
        raise ValueError("Unsupported frozen P4 gate schema/protocol")
    gate._parameters(gs["parameters"])
    s = gs["frozen_base_parameters"]
    required = {"mode", "interface_scale", "activation_energy_eV", "gem_coefficients",
                "domain_coefficients", "gem_fixed"}
    if not isinstance(s, Mapping) or set(s) != required:
        raise ValueError("Exact frozen base parameter binding required")
    names = domain._physics.FEATURE_NAMES+("hopping_strength_log10",)
    coef = _vector(s["gem_coefficients"], names, "frozen GEM")
    beta = _vector(s["domain_coefficients"], domain.PARAMETER_NAMES, "frozen domain")
    gem_summary = ps["gem"]
    if (not isinstance(gem_summary["coefficients"], Mapping)
            or not isinstance(ps["domain_parameters"], Mapping)
            or not np.array_equal(coef, _vector(gem_summary["coefficients"], names, "summary GEM"))
            or not np.array_equal(beta, _vector(ps["domain_parameters"], domain.PARAMETER_NAMES, "summary domain"))):
        raise ValueError("Physics summary and gate frozen coefficients differ")
    fixed = s["gem_fixed"]
    if (not isinstance(fixed, Mapping) or set(fixed) != set(FIXED_PRIORS)|{"enable_hopping"}
            or type(fixed["enable_hopping"]) is not bool):
        raise ValueError("Complete fixed GEM geometry/hopping binding required")
    numeric_fixed = {k: _number(fixed[k], k) for k in FIXED_PRIORS}
    for key, prior_key in FIXED_PRIORS.items():
        if numeric_fixed[key] != _number(gem_summary["fixed_priors"][prior_key], prior_key):
            raise ValueError("Physics summary and frozen GEM fixed priors differ")
    if s["mode"] not in domain.MODES or ps["mode"] != s["mode"] or ps["coupling"] not in ("sequential", "joint"):
        raise ValueError("Unsupported/inconsistent frozen domain mode/coupling")
    scale = _number(s["interface_scale"], "interface_scale", nonnegative=True)
    energy = _number(s["activation_energy_eV"], "activation_energy_eV", nonnegative=True)
    if (scale != _number(ps["interface_scale"], "summary interface_scale", nonnegative=True)
            or energy != _number(ps["activation_energy_eV"], "summary activation_energy", nonnegative=True)):
        raise ValueError("Physics summary and gate frozen domain configuration differ")
    maximum = gem_summary["max_nfev"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Explicit positive integer max_nfev required")
    regularization = _number(ps["regularization"], "domain regularization", nonnegative=True)
    gem_regularization = _number(ps["gem_regularization"], "gem regularization", nonnegative=True)
    if gem_regularization != _number(gem_summary["regularization"], "summary gem regularization", nonnegative=True):
        raise ValueError("GEM regularization summaries differ")
    base = domain.DomainTransport(mode=s["mode"], coupling=ps["coupling"], interface_scale=scale,
        activation_energy_eV=energy, max_nfev=maximum, regularization=regularization,
        gem_regularization=gem_regularization)
    base.gem_ = domain._physics.GreyBoxTransport(**numeric_fixed,
        enable_hopping=fixed["enable_hopping"], max_nfev=maximum, regularization=gem_regularization)
    base.gem_.coef_, base.domain_parameters_ = coef.copy(), beta.copy()
    return base


def reference_state(physical_rows, physics_summary, gate_summary, acid_rows, protocol_admitted):
    """Recompute actual gated P4 and its low phase from explicit plain params.

    Contrast is returned unchanged, even outside a future network table's
    domain. The caller must separately admit/reject it; no clipping or inverse.
    This constructs forward objects, not fitted optimizers or an audit receipt.
    """
    physical, acids = list(physical_rows), list(acid_rows)
    if len(physical) != len(acids):
        raise ValueError("Physical/typed acid rows must align")
    if any(not isinstance(r, Mapping) or RECORD_KEYS.intersection(r) for r in physical):
        raise ValueError("Physical rows cannot contain records, membership or query targets")
    if any(not isinstance(r, Mapping) or not set(r).issubset(ACID_KEYS) for r in acids):
        raise ValueError("Only four explicit typed acid input keys are accepted")
    mask = np.asarray(protocol_admitted)
    if mask.dtype != np.dtype(bool) or mask.shape != (len(physical),):
        raise ValueError("Explicit aligned boolean protocol mask required")
    base = _base(physics_summary, gate_summary)
    parts = gate.apply_acid_process_gate(physical, acids, base, gate_summary["parameters"],
        protocol_admitted=mask, protocol_scope=gate_summary["protocol_scope"])
    prepared, design, _, matrix = domain._inputs(physical, base.activation_energy_eV)
    design[:, gate.GEM_ACID_COLUMN] = parts["acid_process_new_coordinate"]
    matrix[:, gate.DOMAIN_ACID_COLUMN] = -parts["acid_process_new_coordinate"]
    hopping, _ = domain._physics.hopping_coordinate(prepared)
    if not base.gem_.enable_hopping:
        hopping[:] = 0.
    gem_log, low, _, _ = base.gem_._compute(design, hopping, base.gem_.coef_)
    n, gem = len(physical), base.gem_
    if low.shape != (n,) or not np.isfinite(low).all() or np.any(low <= 0.) or np.any(low > gem.sigma_high):
        raise ValueError("Actual gated low phase failed positive phase bounds")
    replay = domain.resistor_network(gem_log, matrix, base.domain_parameters_, base.mode, base.interface_scale)
    gem_error = gate._maximum_abs(gem_log-parts["gem_log10"])
    circuit_error = max(gate._maximum_abs(replay[k]-parts[k]) for k in replay)
    residual, violation = gate._check_parts(parts, low, gem)
    if gem_error > 1e-12 or circuit_error > 1e-12 or violation.any():
        raise ValueError("Actual direct state does not bind gated GEM/circuit")
    state = dict(effective_low_phase_S_cm=low.copy(), high_sigma_S_cm=np.full(n, gem.sigma_high),
        occupancy=np.full(n, gem.fraction), contrast=low/gem.sigma_high,
        rho_ohm_cm=parts["series_resistivity_ohm_cm"].copy(), p_bypass=parts["bypass_fraction"].copy())
    state["reference_parts"] = {k: v.copy() for k, v in parts.items()}
    state["checks"] = dict(n_rows=n, direct_gem_max_abs_error=gem_error,
        direct_circuit_max_abs_error=circuit_error,
        gem_equation_residual_max_abs=gate._maximum_abs(residual),
        circuit_closure_residual_max_abs=gate._maximum_abs(parts["closure_residual"]),
        n_phase_bound_violations=int(violation.sum()),
        n_domain_bound_violations=int(parts["domain_bound_violation"].sum()),
        n_acid_coordinate_changed=int(parts["acid_process_coordinate_changed"].sum()))
    state["metadata"] = dict(version=VERSION, gate_version=gate.VERSION,
        protocol_scope=gate_summary["protocol_scope"], enable_hopping=gem.enable_hopping,
        scope="Pure plain-parameter forward, not optimizer restoration, fitting or saved-model replay",
        state_scope="Effective fixed fraction/low phase/rho/bypass, not measured morphology or phase conductivities",
        table_admission="Caller-owned; raw contrast is not clipped into table support",
        reference_scope="All old P4 arrays, including GEM/acid checks, are in reference_parts only")
    return state
