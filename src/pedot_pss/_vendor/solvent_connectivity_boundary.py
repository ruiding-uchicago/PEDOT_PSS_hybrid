"""Pure initial-solvent/first-dry effective GEM connectivity hypothesis.

fc_new=expit(logit(fc_frozen)-eta*(dmso_memory+eg_memory)), eta in [0,3].
This is a shared effective connectivity assumption, not measured morphology,
dry PEDOT volume, a paper-validated constitutive law or identified activation
energy. The reused .35 eV processing prior is explicitly only a sensitivity
assumption. Caller must verify stage/protocol admission; this module never
infers it from identity, source, outcomes or electrical measurements.
No files, fitting, inverse labels, mutable model state or additive sigma patch.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import expit, logit

try:
    from . import solvent_drying_boundary as boundary
except ImportError:
    from . import solvent_drying_boundary as boundary
VERSION = "initial-solvent-first-dry-GEM-connectivity-v1"
CHECK_TOLERANCE = 1e-10
CORE_KEYS = frozenset(("prediction_log10_S_cm", "gem_log10", "domain_correction_log10",
    "series_correction_log10", "sigma_domain_S_cm", "sigma_gem_S_cm", "sigma_series_S_cm",
    "series_resistivity_ohm_cm", "barrier_rho_ohm_cm", "anisotropy_factor", "bypass_fraction",
    "closure_residual", "domain_bound_violation", "gem_equation_residual", "gem_phase_bound_violation"))
ACID_TRACE_KEYS = frozenset("acid_process_"+key for key in (
    "protocol_admitted", "has_supported_concentration", "has_neat_proxy", "saturation", "hydration_factor", "gate",
    "is_admitted", "coordinate_changed", "old_coordinate", "new_coordinate", "base_prediction_log10_S_cm",
    "base_gem_log10", "correction_log10", "gem_correction_log10", "old_log_rho_linear", "new_log_rho_linear"))


def geometry(gem_model):
    """Actual frozen GEM geometry; f may equal/exceed fc without clipping."""
    if type(gem_model) is not boundary.transport._physics.GreyBoxTransport:
        raise TypeError("Actual GreyBoxTransport GEM geometry required")
    values = (gem_model.sigma_high, gem_model.fraction, gem_model.threshold, gem_model.exponent)
    if (any(not boundary._finite(v) for v in values) or values[0] <= 0. or
            not 0. <= values[1] <= 1. or not 0. < values[2] < 1. or values[3] <= 0.):
        raise ValueError("Finite positive GEM geometry and valid fractions required")
    return tuple(float(v) for v in values)


def _forward(low, config, threshold, reference):
    high, fraction, _, exponent = config
    physics = boundary.transport._physics
    g = physics.gem_conductivity(low, high, fraction, threshold, exponent)
    out = boundary.network_with_rho(np.log10(g), reference["series_resistivity_ohm_cm"],
        reference["bypass_fraction"], anisotropy_factor=reference["anisotropy_factor"])
    residual = physics.gem_residual(out["sigma_gem_S_cm"], low, high, fraction, threshold, exponent)
    violation = ((out["sigma_gem_S_cm"] < low*(1.-1e-11)) |
                 (out["sigma_gem_S_cm"] > high*(1.+1e-11))).astype(float)
    if not np.isfinite(residual).all() or (abs(residual) > CHECK_TOLERANCE).any() or violation.any():
        raise ValueError("Actual GEM equation/phase bounds failed")
    out.update(gem_equation_residual=residual, gem_phase_bound_violation=violation)
    return out


def _reference(parts, low, config):
    if not isinstance(parts, Mapping) or not CORE_KEYS.issubset(parts):
        raise ValueError("Complete actual reference GEM/circuit core required")
    if not set(parts).issubset(CORE_KEYS | ACID_TRACE_KEYS):
        raise ValueError("Only physical/reference trace fields accepted; no targets or IDs")
    ref = {key: boundary._array(value, len(low), key) for key, value in parts.items()}
    conductivity_keys = ("sigma_domain_S_cm", "sigma_gem_S_cm", "sigma_series_S_cm")
    if any((ref[key] <= 0.).any() for key in conductivity_keys):
        raise ValueError("Positive reference conductivities required")
    if not np.array_equal(ref["barrier_rho_ohm_cm"], ref["series_resistivity_ohm_cm"]):
        raise ValueError("Reference effective rho aliases differ")
    for key in ("gem_equation_residual", "closure_residual"):
        if (abs(ref[key]) > CHECK_TOLERANCE).any(): raise ValueError("Reference equation failed")
    for key in ("gem_phase_bound_violation", "domain_bound_violation"):
        if ref[key].any(): raise ValueError("Reference physical bounds failed")
    expected = _forward(low, config, np.full(len(low), config[2]), ref)
    for key in CORE_KEYS:
        atol = 0. if key in conductivity_keys else CHECK_TOLERANCE
        if not np.allclose(ref[key], expected[key], rtol=CHECK_TOLERANCE, atol=atol):
            raise ValueError("Reference inconsistent with supplied low/GEM/circuit: " + key)
    return ref


def apply_solvent_connectivity_boundary(low_phase_S_cm, gem_model, reference_parts,
                                        drying_rows, *, eta, protocol_admitted):
    """Re-solve actual GEM with a recipe-dependent fc and fixed effective rho/p.

    low_phase_S_cm is the explicit actual input-state vector, 0<low<=high.
    Reference core (all15 fields) must match this vector and frozen geometry.
    rho ALREADY includes interface scale/anisotropy; neither is multiplied again.
    Only admitted initial vol%/first-dry coordinates can change fc. Missing or
    conflicting inputs remain exact reference, not fabricated zero conditions.
    eta0, unchanged floating fc and threshold-independent pure/equal phases
    copy the old core exactly. rho0/p1 still permit an actual GEM fc response.
    Historical acid traces are copied ONLY with solvent_connectivity_reference_
    prefix; they are not recomputed gate effects at the new connectivity.
    """
    if not boundary._finite(eta) or not 0. <= eta <= 3.:
        raise ValueError("eta must be finite, non-Boolean and within [0,3]")
    rows = list(drying_rows)
    low = boundary._array(low_phase_S_cm, len(rows), "effective low phase S/cm")
    config = geometry(gem_model)
    if ((low <= 0.) | (low > config[0])).any():
        raise ValueError("Positive low phase not exceeding frozen high phase required")
    ref = _reference(reference_parts, low, config)
    coords = boundary.drying_coordinates(rows, protocol_admitted=protocol_admitted)
    memory = coords["solvent_drying_dmso_memory"] + coords["solvent_drying_eg_memory"]
    threshold = np.full(len(rows), config[2])
    active = (coords["solvent_drying_is_admitted"] == 1.) & (memory > 0.) & (eta > 0.)
    base_logit = logit(config[2])
    shifted_logit = base_logit-float(eta)*memory
    representable = active & (shifted_logit != base_logit)
    threshold[representable] = expit(shifted_logit[representable])
    if ((threshold <= 0.) | (threshold >= 1.)).any():
        raise ValueError("Modified threshold lost positive finite geometry")
    changed = threshold != config[2]
    recomputed = changed & (low != config[0]) & (0. < config[1] < 1.)
    result = {key: ref[key].copy() for key in CORE_KEYS}
    if recomputed.any():
        corrected = _forward(low, config, threshold, ref)
        for key in CORE_KEYS: result[key][recomputed] = corrected[key][recomputed]
    result.update(coords)
    result.update({"solvent_connectivity_reference_"+key: value.copy() for key, value in ref.items()})
    result.update(solvent_connectivity_base_threshold=np.full(len(rows), config[2]),
        solvent_connectivity_threshold=threshold, solvent_connectivity_threshold_changed=changed.astype(float),
        solvent_connectivity_recomputed=recomputed.astype(float),
        solvent_connectivity_eta=np.full(len(rows), float(eta)),
        solvent_connectivity_correction_log10=result["prediction_log10_S_cm"]-ref["prediction_log10_S_cm"],
        solvent_connectivity_gem_correction_log10=result["gem_log10"]-ref["gem_log10"])
    return result
