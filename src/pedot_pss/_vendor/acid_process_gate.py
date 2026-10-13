"""Pure forward replacement of the frozen GEM/circuit acid process coordinate.

This is an effective, protocol-limited initial-solution gate, NOT measured PSS
loss, water activity, dissociation or retained film water. It does not fit, read
files/IDs/targets, load models or change the supplied base. Physical rows and
typed acid rows are deliberately separate: only physical rows enter transport.
The caller must independently verify the declared hotdrop/dry/rinse protocol.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

try:
    from . import acid_interface_closure as acid
    from . import domain_transport_v2 as transport
except ImportError:
    from . import acid_interface_closure as acid
    from . import domain_transport_v2 as transport
VERSION = "acid-process-coordinate-gate-v1"
PROTOCOL_SCOPE = acid.PROTOCOL_SCOPE
INPUT_VERSION = acid.INPUT_VERSION
PARAMETER_BOUNDS = {"eta": (0., 1.),
                    "half_saturation_mol_L": (.01, 100.),
                    "hydration_exponent": (.1, 8.)}
CHECK_TOLERANCE = 1e-10
GEM_ACID_COLUMN = 3
DOMAIN_ACID_COLUMN = 5


def _maximum_abs(value):
    return float(np.max(abs(value))) if len(value) else 0.


def _parameters(parameters):
    if not isinstance(parameters, Mapping) or set(parameters) != set(PARAMETER_BOUNDS):
        raise ValueError("Exactly eta, half_saturation_mol_L and hydration_exponent required")
    result = {}
    for key, (low, high) in PARAMETER_BOUNDS.items():
        value = parameters[key]
        if not acid._finite_number(value) or not low <= value <= high:
            raise ValueError("Parameter outside finite declared bounds: " + key)
        result[key] = float(value)
    return result


def process_gate_coordinates(acid_rows, parameters, *, protocol_admitted,
                             protocol_scope=None):
    """Input-only gate; missing/unknown/out-of-range inputs stay unavailable.

    All six supported acids share c/(K+c). Only the three named monoacids
    receive the neat-endpoint proximity factor. Its ratio is NOT water mole
    fraction. Explicit zero c is known (gate zero), not imputed missing c.
    Scope None is inactive; an unrecognized claimed scope is rejected.
    """
    rows, pars = list(acid_rows), _parameters(parameters)
    protocol = np.asarray(protocol_admitted)
    if protocol.shape != (len(rows),) or protocol.dtype != np.dtype(bool):
        raise ValueError("protocol_admitted must be an aligned Boolean vector")
    if protocol_scope not in (None, PROTOCOL_SCOPE):
        raise ValueError("Unrecognized verified acid protocol scope")
    protocol = protocol.copy() if protocol_scope == PROTOCOL_SCOPE else np.zeros(len(rows), bool)
    coords = acid.acid_coordinates(rows)
    supported = np.asarray([c["has_concentration"] for c in coords], bool)
    saturation, hydration, gate = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros(len(rows))
    neat = np.zeros(len(rows))
    for i, coord in enumerate(coords):
        if not protocol[i] or not supported[i]:
            continue
        c = coord["concentration_mol_L"]
        saturation[i] = c / (pars["half_saturation_mol_L"] + c)
        hydration[i] = 1.
        if coord["has_neat_proxy"]:
            neat[i] = 1.
            q = coord["neat_ratio"]
            # Explicit endpoint avoids log(0), 0**0 and invalid Jacobians.
            hydration[i] = 0. if q == 1. else math.exp(pars["hydration_exponent"] * math.log1p(-q))
        gate[i] = saturation[i] * hydration[i]
    if not np.isfinite(gate).all() or ((gate < 0.) | (gate > 1.)).any():
        raise ValueError("Acid gate outside finite [0,1]")
    return dict(acid_process_protocol_admitted=protocol.astype(float),
                acid_process_has_supported_concentration=supported.astype(float),
                acid_process_has_neat_proxy=neat, acid_process_saturation=saturation,
                acid_process_hydration_factor=hydration, acid_process_gate=gate)


def _array(value, n, name):
    result = np.asarray(value, float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError("Nonfinite/misaligned actual physical component: " + name)
    return result.copy()


def _check_parts(parts, low_phase, gem):
    n = len(low_phase)
    for name, value in parts.items():
        _array(value, n, name)
    sigma = parts["sigma_gem_S_cm"]
    violation = ((sigma < low_phase*(1.-1e-11)) |
                 (sigma > gem.sigma_high*(1.+1e-11))).astype(float)
    residual = transport._physics.gem_residual(sigma, low_phase, gem.sigma_high,
                                              gem.fraction, gem.threshold, gem.exponent)
    if (not np.isfinite(residual).all() or _maximum_abs(residual) > CHECK_TOLERANCE or
            _maximum_abs(parts["closure_residual"]) > CHECK_TOLERANCE or
            violation.any() or parts["domain_bound_violation"].any() or
            (parts["sigma_domain_S_cm"] <= 0).any() or (low_phase <= 0).any()):
        raise ValueError("Actual GEM phase/equation or resistor-network check failed")
    return residual, violation


def apply_acid_process_gate(physical_rows, acid_rows, base, parameters, *,
                            protocol_admitted, protocol_scope=None):
    """Re-solve GEM and the complete circuit with frozen coefficients.

    a_new = a_old * ((1-eta) + eta*g(c)). The real GEM acid column 3 and
    negative circuit acid column 5 change together, ONLY if both original
    columns certify acid=1 and the typed concentration/protocol are supported.
    Every other coordinate (including thermal exposure, hopping and bypass)
    and every old coefficient remains unchanged. eta=0 and inactive rows copy
    ALL standard C0 components exactly. No additive log-conductivity patch.

    eta=1 at a neat monoacid endpoint switches off the old *acid coordinate*,
    not all processing history and not necessarily the untreated conductivity.
    Train-only estimation of the three parameters is the future caller's job.
    """
    physical_rows, acid_rows = list(physical_rows), list(acid_rows)
    if len(physical_rows) != len(acid_rows):
        raise ValueError("Separate physical and acid rows must align")
    pars = _parameters(parameters)
    coords = process_gate_coordinates(acid_rows, pars, protocol_admitted=protocol_admitted,
                                     protocol_scope=protocol_scope)
    if type(base) is not transport.DomainTransport or type(getattr(base, "gem_", None)) is not transport._physics.GreyBoxTransport:
        raise TypeError("An actual frozen DomainTransport/GreyBoxTransport base is required")
    gem = base.gem_
    names = transport._physics.FEATURE_NAMES
    if names[GEM_ACID_COLUMN] != "acid_post_present" or transport.PARAMETER_NAMES[DOMAIN_ACID_COLUMN] != "acid_barrier_relief":
        raise ValueError("Frozen physical acid column schema changed")
    coef = np.asarray(gem.coef_, float).copy()
    beta = np.asarray(getattr(base, "domain_parameters_", None), float).copy()
    if (coef.shape != (len(names)+1,) or beta.shape != (10,) or
            not np.isfinite(coef).all() or not np.isfinite(beta).all()):
        raise ValueError("Complete finite unchanged GEM/domain coefficients required")
    if (base.mode not in transport.MODES or not acid._finite_number(base.interface_scale) or base.interface_scale < 0 or
            not acid._finite_number(base.activation_energy_eV) or base.activation_energy_eV < 0):
        raise ValueError("Invalid frozen transport configuration")
    # The ordinary base path is authoritative; never trust caller-supplied
    # domain_acid_post_present or cached coordinate fields in acid_rows.
    n = len(physical_rows)
    old = {key: _array(value, n, key) for key, value in base.components(physical_rows).items()}
    prepared, design, domain_coords, matrix = transport._inputs(physical_rows, base.activation_energy_eV)
    design, matrix = np.asarray(design, float), np.asarray(matrix, float)
    if (design.shape != (n, len(names)) or matrix.shape != (n, 9) or
            not np.isfinite(design).all() or not np.isfinite(matrix).all()):
        raise ValueError("Actual physical design dimensions/values changed")
    actual_acid = design[:, GEM_ACID_COLUMN]
    if (((actual_acid != 0.) & (actual_acid != 1.)).any() or
            not np.array_equal(matrix[:, DOMAIN_ACID_COLUMN], -actual_acid) or
            not np.array_equal(np.asarray([r[acid.OLD_ACID_FLAG_KEY] for r in domain_coords]), actual_acid)):
        raise ValueError("Actual GEM and circuit acid coordinates are inconsistent")
    hopping, _ = transport._physics.hopping_coordinate(prepared)
    if not gem.enable_hopping:
        hopping[:] = 0.
    old_gem, _, _, _ = gem._compute(design, hopping, coef)
    if _maximum_abs(old_gem-old["gem_log10"]) > 1e-12:
        raise ValueError("Frozen base GEM does not bind its actual process design")
    admitted = ((coords["acid_process_protocol_admitted"] == 1.) &
                (coords["acid_process_has_supported_concentration"] == 1.) & (actual_acid == 1.))
    new_acid = actual_acid.copy()
    new_acid[admitted] *= (1.-pars["eta"]) + pars["eta"]*coords["acid_process_gate"][admitted]
    if ((new_acid < 0.) | (new_acid > 1.)).any():
        raise ValueError("Replacement acid coordinate outside [0,1]")
    new_design, new_matrix = design.copy(), matrix.copy()
    new_design[:, GEM_ACID_COLUMN], new_matrix[:, DOMAIN_ACID_COLUMN] = new_acid, -new_acid
    prediction, low_phase, _, score = gem._compute(new_design, hopping.copy(), coef.copy())
    for key, value in (("GEM prediction", prediction), ("low phase", low_phase), ("process score", score)):
        _array(value, n, key)
    result = transport.resistor_network(prediction, new_matrix, beta.copy(), base.mode, base.interface_scale)
    residual, violation = _check_parts(result, low_phase, gem)
    result["gem_equation_residual"], result["gem_phase_bound_violation"] = residual, violation
    changed = new_acid != actual_acid
    for key in old:
        result[key][~changed] = old[key][~changed]
    if not np.array_equal(coef, gem.coef_) or not np.array_equal(beta, base.domain_parameters_):
        raise ValueError("Frozen coefficients changed during forward evaluation")
    result.update(coords)
    result.update(acid_process_is_admitted=admitted.astype(float),
        acid_process_coordinate_changed=changed.astype(float),
        acid_process_old_coordinate=actual_acid.copy(), acid_process_new_coordinate=new_acid,
        acid_process_base_prediction_log10_S_cm=old["prediction_log10_S_cm"].copy(),
        acid_process_base_gem_log10=old["gem_log10"].copy(),
        acid_process_correction_log10=result["prediction_log10_S_cm"]-old["prediction_log10_S_cm"],
        acid_process_gem_correction_log10=result["gem_log10"]-old["gem_log10"],
        acid_process_old_log_rho_linear=matrix[:, :7] @ beta[:7],
        acid_process_new_log_rho_linear=new_matrix[:, :7] @ beta[:7])
    return result
