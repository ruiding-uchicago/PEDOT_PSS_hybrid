"""Pure first-film-drying memory closure; no fit, files, targets or state writes.

Only caller-verified initial DMSO/EG vol% and first-dry C/seconds are read.
Protocol/stage admission belongs to the caller, never to source IDs. Missing
channels are flagged unknown, not asserted zero. H is an effective Arrhenius
processing prior, not retained solvent, water activity or measured morphology.
The ALREADY-clipped base rho is multiplied by exp(delta); no second clip.
Corrected rho may exceed the old exp(30) guard, which is not a physical bound.
All phase parameters, bypass, anisotropy and interface_scale remain frozen.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import numbers

import numpy as np

try:
    from . import domain_transport_v2 as transport
except ImportError:
    from . import domain_transport_v2 as transport
VERSION = "initial-solvent-first-drying-boundary-v1"
PARAMETER_BOUNDS = {"A_D": (0., 4.), "A_E": (0., 4.), "B_D": (0., 4.)}
DOSE_KEYS = {"dmso": ("recipe_dmso_dose_volpct", "pb_dmso_volpct_reported"),
             "eg": ("recipe_eg_dose_volpct", "pb_eg_volpct_reported")}
TEMPERATURE_KEY = "first_dry_temperature_C"
DURATION_KEY = "first_dry_duration_s"
ACTIVATION_ENERGY_EV, BOLTZMANN_EV_K = .35, 8.617333262e-5
REFERENCE_TEMPERATURE_K, REFERENCE_DURATION_S = 403.15, 60.
CHECK_TOLERANCE = 1e-10


def _finite(value):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError, TypeError):
        return False


def _parameters(parameters):
    if not isinstance(parameters, Mapping) or set(parameters) != set(PARAMETER_BOUNDS):
        raise ValueError("Exactly A_D, A_E, B_D parameters required")
    if any(not _finite(parameters[k]) or not lo <= parameters[k] <= hi
           for k, (lo, hi) in PARAMETER_BOUNDS.items()):
        raise ValueError("Parameters must be finite, non-Boolean and within [0,4]")
    return {k: float(parameters[k]) for k in PARAMETER_BOUNDS}


def _array(value, n, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (n,):
        raise ValueError("Aligned numeric vector required: " + name)
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Finite vector required: " + name)
    return result


def drying_coordinates(rows, *, protocol_admitted):
    """Controlled input-only coordinates; malformed supplied doses disable row.

    Missing one channel disables that contribution only; at least one explicit
    channel and valid verified first-dry T/time are required. Alias conflicts
    (including zero versus tiny positive) are never resolved by precedence.
    No wt%, generic dry-stage aliases, strings or Celsius/Kelvin guessing.
    """
    rows = list(rows)
    protocol = np.asarray(protocol_admitted)
    if protocol.shape != (len(rows),) or protocol.dtype != np.dtype(bool):
        raise ValueError("protocol_admitted must be an aligned Boolean vector")
    keys = ("protocol_admitted", "has_first_dry", "has_dmso_volpct", "has_eg_volpct",
            "is_admitted", "thermal_saturation", "dmso_memory", "eg_memory")
    out = {"solvent_drying_" + k: np.zeros(len(rows)) for k in keys}
    out["solvent_drying_protocol_admitted"] = protocol.astype(float)
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        doses, malformed = {}, False
        for species, aliases in DOSE_KEYS.items():
            supplied = [row[k] for k in aliases if k in row]
            valid = bool(supplied) and all(_finite(v) and 0. <= v <= 100. for v in supplied)
            if valid and any(v != supplied[0] for v in supplied[1:]):
                valid = False
            malformed |= bool(supplied) and not valid
            out["solvent_drying_has_" + species + "_volpct"][i] = float(valid)
            doses[species] = float(supplied[0]) if valid else None
        temp, seconds = row.get(TEMPERATURE_KEY), row.get(DURATION_KEY)
        valid_dry = (_finite(temp) and temp > -273.15 and _finite(seconds) and seconds >= 0.)
        out["solvent_drying_has_first_dry"][i] = float(valid_dry)
        if not protocol[i] or malformed or not valid_dry or all(v is None for v in doses.values()):
            continue
        h = 0.
        if seconds > 0.:
            log_e = (math.log(seconds) - math.log(REFERENCE_DURATION_S) -
                     ACTIVATION_ENERGY_EV / BOLTZMANN_EV_K *
                     (1. / (temp + 273.15) - 1. / REFERENCE_TEMPERATURE_K))
            h = 1. / (1. + math.exp(-log_e)) if log_e >= 0. else math.exp(log_e) / (1. + math.exp(log_e))
        out["solvent_drying_is_admitted"][i] = 1.
        out["solvent_drying_thermal_saturation"][i] = h
        for species, dose in doses.items():
            if dose is not None:
                out["solvent_drying_" + species + "_memory"][i] = dose / (5. + dose) * h
    return out


def network_with_rho(gem_log10, rho_ohm_cm, bypass_fraction, *, anisotropy_factor=None):
    """Same positive series/bypass circuit with an explicit effective rho.

    rho already includes frozen interface_scale and anisotropy; neither is
    multiplied again. Fractions 0 and 1 have exact limiting interpretations.
    This helper is not a fit, a phase-fraction inference or an output patch.
    """
    n = len(gem_log10)
    gem, rho, p = (_array(v, n, name) for v, name in
                   ((gem_log10, "GEM"), (rho_ohm_cm, "rho"), (bypass_fraction, "bypass")))
    anisotropy = np.ones(n) if anisotropy_factor is None else _array(anisotropy_factor, n, "anisotropy")
    if (rho < 0.).any() or ((p < 0.) | (p > 1.)).any() or (anisotropy <= 0.).any():
        raise ValueError("Positive rho/anisotropy and [0,1] bypass required")
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        g = 10. ** gem
        loading = rho * g
        series_correction = -np.log1p(loading) / transport.LN10
        series = g / (1. + loading)
        correction = np.logaddexp(np.log(p), np.log1p(-p) - np.log1p(loading)) / transport.LN10
        correction[p == 0.] = series_correction[p == 0.]
        correction[(p == 1.) | (rho == 0.)] = 0.
        prediction = gem + correction
        sigma = 10. ** prediction
        expected = p * g + (1. - p) * series
        residual = (sigma - expected) / np.maximum(expected, 1e-300)
    bounds = ((sigma < series * (1. - 1e-11)) | (sigma > g * (1. + 1e-11))).astype(float)
    result = dict(prediction_log10_S_cm=prediction, gem_log10=gem,
        domain_correction_log10=correction, series_correction_log10=series_correction,
        sigma_domain_S_cm=sigma, sigma_gem_S_cm=g, sigma_series_S_cm=series,
        series_resistivity_ohm_cm=rho, barrier_rho_ohm_cm=rho.copy(),
        anisotropy_factor=anisotropy, bypass_fraction=p, closure_residual=residual,
        domain_bound_violation=bounds)
    if (any(not np.isfinite(v).all() for v in result.values()) or (g <= 0.).any() or
            (sigma <= 0.).any() or (series <= 0.).any() or bounds.any() or
            (abs(residual) > CHECK_TOLERANCE).any()):
        raise ValueError("Finite positive phase/circuit solution required")
    return result


def apply_solvent_drying_boundary(physical_rows, drying_rows, base, parameters, *, protocol_admitted):
    """Actual frozen GEM plus modified rho before the unchanged circuit.

    All-zero parameters, unsupported rows, gem_only and interface_scale=0
    copy every old component exactly. New trace fields describe this physical
    closure, not an additive ML prediction or a microscopic identification.
    """
    physical_rows, drying_rows = list(physical_rows), list(drying_rows)
    if len(physical_rows) != len(drying_rows):
        raise ValueError("Physical/drying rows must align")
    pars = _parameters(parameters)
    coords = drying_coordinates(drying_rows, protocol_admitted=protocol_admitted)
    if type(base) is not transport.DomainTransport or type(getattr(base, "gem_", None)) is not transport._physics.GreyBoxTransport:
        raise TypeError("Actual frozen DomainTransport/GreyBoxTransport required")
    gem, n = base.gem_, len(physical_rows)
    coef, beta = np.asarray(gem.coef_, float).copy(), np.asarray(base.domain_parameters_, float).copy()
    if (coef.shape != (len(transport._physics.FEATURE_NAMES)+1,) or beta.shape != (10,) or
            not np.isfinite(coef).all() or not np.isfinite(beta).all() or base.mode not in transport.MODES or
            not _finite(base.interface_scale) or base.interface_scale < 0. or
            not _finite(base.activation_energy_eV) or base.activation_energy_eV < 0.):
        raise ValueError("Complete finite frozen GEM/domain configuration required")
    if (any(not _finite(v) for v in (gem.sigma_high, gem.fraction, gem.threshold, gem.exponent, gem.low_log10_min)) or
            gem.sigma_high <= 0. or not 0. <= gem.fraction < gem.threshold < 1. or gem.exponent <= 0. or
            gem.low_log10_min >= math.log10(gem.sigma_high) or type(gem.enable_hopping) is not bool):
        raise ValueError("Invalid frozen GEM phase configuration")
    old = {k: _array(v, n, k) for k, v in base.components(physical_rows).items()}
    prepared = transport._inputs(physical_rows, base.activation_energy_eV)[0]
    actual_gem = gem.components(prepared)
    low, g = actual_gem["effective_low_phase_S_cm"], old["sigma_gem_S_cm"]
    residual = transport._physics.gem_residual(g, low, gem.sigma_high, gem.fraction, gem.threshold, gem.exponent)
    if (not np.array_equal(actual_gem["prediction_log10_S_cm"], old["gem_log10"]) or
            not np.isfinite(residual).all() or (abs(residual) > CHECK_TOLERANCE).any() or
            (abs(old["closure_residual"]) > CHECK_TOLERANCE).any() or
            old["domain_bound_violation"].any() or old["gem_phase_bound_violation"].any()):
        raise ValueError("Actual frozen GEM/circuit equations failed")
    d, e = coords["solvent_drying_dmso_memory"], coords["solvent_drying_eg_memory"]
    delta = -pars["A_D"]*d - pars["A_E"]*e + pars["B_D"]*d*d
    with np.errstate(over="ignore", under="ignore"):
        rho = old["series_resistivity_ohm_cm"] * np.exp(delta)
    if ((old["series_resistivity_ohm_cm"] > 0.) & (rho == 0.)).any():
        raise ValueError("Positive corrected rho underflowed")
    corrected = network_with_rho(old["gem_log10"], rho, old["bypass_fraction"],
                                 anisotropy_factor=old["anisotropy_factor"])
    changed = rho != old["series_resistivity_ohm_cm"]
    result = {k: v.copy() for k, v in old.items()}
    for key, value in corrected.items():
        result[key][changed] = value[changed]
    if not np.array_equal(coef, gem.coef_) or not np.array_equal(beta, base.domain_parameters_):
        raise ValueError("Frozen state changed during forward evaluation")
    result.update(coords)
    result.update(solvent_drying_delta_ln_rho=delta, solvent_drying_rho_changed=changed.astype(float),
        solvent_drying_base_prediction_log10_S_cm=old["prediction_log10_S_cm"].copy(),
        solvent_drying_base_rho_ohm_cm=old["series_resistivity_ohm_cm"].copy(),
        solvent_drying_correction_log10=result["prediction_log10_S_cm"]-old["prediction_log10_S_cm"])
    return result
