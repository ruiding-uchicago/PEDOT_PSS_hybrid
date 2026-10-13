"""Reviewed PEO/PH1000-EG outer transport, not internal PEDOT volume mixing.

Author dry polymer masses and assumed densities define NOMINAL volume fractions;
retained EG/water and excess volumes are unknown. The normalized percolation law
is a reparameterization of the reported power law; outer GEM is our extension.
The low phase 1e-12 S/cm is a model prior, not measured PEO conductivity. Fitted
parameters use caller-supplied TRAIN labels only; split governance is the caller's
responsibility. No IO, IDs, outcome-based admission or whole-paper fit priors.
"""
import math
import numbers
from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares

from .hybrid import _physics

VERSION = "reviewed-peo-outer-transport-v1"
LAWS = ("percolation", "gem")
DENSITIES = (1.55, 1.12)  # g/cm3, author-assumed host and PEO density priors
SIGMA_LOW_S_CM = 1e-12
PARAMETER_KEYS = ("log10_sigma_host", "fc", "t")
BOUNDS = ((-6., .005, .3), (math.log10(2e4), .25, 3.))
STARTS = ((2., .03, 1.), (3., .06, 1.), (4., .12, 2.))
SCOPE = ("Nominal dry PEO/PH1000-EG outer composite; not measured final volume "
         "fraction or internal PEDOT fraction. Complete material expert, no inherited "
         "PEDOT residual. Posthoc branch development, not new/nested validation.")


def _number(value):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _vector(value, name):
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in "iuf" or
            any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object))):
        raise ValueError(name + " must be a numeric non-Boolean vector")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def condition(typed_facts, common_typed_facts, processing_profile):
    """Closed factual admission; profile identifiers and row IDs are not read.

    Unknown/conflicting facts stay inactive. Caller authenticates the reviewed
    source-to-specimen join. The two host assays have no PEO dissolving-water.
    """
    result = dict(active=False, phi_nominal=None, reason="malformed_or_missing_facts", scope=SCOPE)
    if not all(isinstance(v, Mapping) for v in (typed_facts, common_typed_facts, processing_profile)):
        return result
    facts, common, profile = typed_facts, common_typed_facts, processing_profile
    required_common = dict(host_identity="PEDOT:PSS", host_grade="PH1000",
        mass_fraction_basis="author_calculated_dry_PEDOT_PSS_mass_over_nominal_total_dry_polymer_film_mass",
        stage="as_fabricated_unwashed", elapsed_storage_duration_s=None,
        explicitly_six_month_stored_branch=False)
    for key, expected in required_common.items():
        if key not in common or type(common[key]) is not type(expected) or common[key] != expected:
            return dict(result, reason="unsupported_common_material_basis_or_stage")
    categorical = dict(host_grade="PH1000", deposition="drop_casting", substrate="Kapton polyimide foil",
        EG_dose_basis="5.25mL per100mL stock; not asserted as final-mixture volume percent",
        PEO_mixing_sonication_applies_only_if_PEO_present=True,
        postwash=False, acid_post_treatment=False, extrusion=False, thermocompression=False)
    for key, expected in categorical.items():
        if key not in profile or type(profile[key]) is not type(expected) or profile[key] != expected:
            return dict(result, reason="unsupported_preparation_profile")
    numeric = dict(stock_solids_wt_percent=1.2, EG_volume_mL=5.25, PEDOT_PSS_stock_volume_mL=100.,
        deposited_solution_volume_uL=800., dry_temperature_C=70., dry_duration_s=2700.,
        anneal_temperature_C=120., anneal_duration_s=3600.)
    if any(not _number(profile.get(k)) or float(profile[k]) != v for k, v in numeric.items()):
        return dict(result, reason="unsupported_preparation_quantity")
    w, phi_given = facts.get("reported_dry_PEDOT_PSS_mass_fraction"), facts.get("nominal_outer_host_volume_fraction")
    if not _number(w) or not 0 < w <= 1 or not _number(phi_given):
        return dict(result, reason="unknown_or_invalid_dry_fraction")
    uncertainty = facts.get("reported_mass_fraction_uncertainty")
    if uncertainty is not None and (not _number(uncertainty) or not 0 <= uncertainty <= 1):
        return dict(result, reason="invalid_reported_uncertainty")
    phi = (w / DENSITIES[0]) / (w / DENSITIES[0] + (1 - w) / DENSITIES[1])
    if abs(float(phi_given) - phi) > 1e-12:
        return dict(result, reason="nominal_density_conversion_conflict")
    if w == 1:
        expected = dict(material_role="pure_EG_host_control", has_PEO_matrix=False,
            matrix_identity=None, matrix_Mw_g_mol=None, PEO_solution_water=None,
            water_not_applicable_no_PEO_solution=True)
    else:
        expected = dict(material_role="PEO_matrix_PEDOT_PSS_EG_composite", has_PEO_matrix=True,
            matrix_identity="PEO", PEO_solution_water="demineralized",
            water_not_applicable_no_PEO_solution=False)
        if (not _number(facts.get("matrix_Mw_g_mol")) or facts["matrix_Mw_g_mol"] != 200000 or
                not _number(profile.get("PEO_mixing_sonication_duration_s")) or
                profile["PEO_mixing_sonication_duration_s"] != 2700):
            return dict(result, reason="unsupported_peo_matrix_or_mixing")
    if any(k not in facts or type(facts[k]) is not type(v) or facts[k] != v for k, v in expected.items()):
        return dict(result, reason="unsupported_matrix_water_or_host_role")
    return dict(result, active=True, phi_nominal=float(phi), reason="reviewed_outer_composite_condition")


def forward(phi, *, law, log10_sigma_host, fc, t):
    """Actual positive outer conductivity in S/cm; empty query arrays are valid."""
    if not isinstance(law, str) or law not in LAWS:
        raise ValueError("Unknown outer transport law")
    parameters = (log10_sigma_host, fc, t)
    if any(not _number(v) or not lo <= v <= hi for v, lo, hi in zip(parameters, *BOUNDS)):
        raise ValueError("Finite non-Boolean parameters within declared bounds required")
    fraction = _vector(phi, "phi")
    if np.any((fraction < 0) | (fraction > 1)):
        raise ValueError("Nominal outer fraction must be in [0,1]")
    host = 10. ** log10_sigma_host
    if law == "gem":
        sigma = np.asarray(_physics.gem_conductivity(SIGMA_LOW_S_CM, host, fraction, fc, t))
        residual = np.asarray(_physics.gem_residual(sigma, SIGMA_LOW_S_CM, host, fraction, fc, t))
    else:
        sigma = np.full(fraction.shape, SIGMA_LOW_S_CM)
        above = fraction > fc
        sigma[above] = np.maximum(SIGMA_LOW_S_CM, host * ((fraction[above] - fc) / (1 - fc)) ** t)
        residual = np.zeros_like(sigma)  # explicit algebraic law, not a GEM residual
    if not np.isfinite(sigma).all() or np.any(sigma < SIGMA_LOW_S_CM) or np.any(sigma > host):
        raise ValueError("Outer conductivity violates positive phase bounds")
    return dict(prediction_log10_S_cm=np.log10(sigma), conductivity_S_cm=sigma,
        equation_residual=residual, metadata=dict(version=VERSION, law=law,
        sigma_low_S_cm=SIGMA_LOW_S_CM, host_role="already_composite_PH1000_EG", scope=SCOPE))


def fit(phi_train, y_train, *, law):
    """Three fixed starts, log10 TRAIN loss only; caller supplies split-safe data."""
    phi, y = _vector(phi_train, "phi_train"), _vector(y_train, "y_train")
    if phi.shape != y.shape or len(y) < 4 or len(np.unique(phi)) < 3:
        raise ValueError("Need aligned >=4 TRAIN rows and >=3 distinct nominal fractions")
    forward(phi, law=law, **dict(zip(PARAMETER_KEYS, STARTS[0])))
    records = []
    for start in STARTS:
        solve = least_squares(lambda v: forward(phi, law=law, **dict(zip(PARAMETER_KEYS, v)))[
            "prediction_log10_S_cm"] - y, start, bounds=BOUNDS, max_nfev=200)
        if not np.isfinite(solve.x).all() or not math.isfinite(float(solve.cost)):
            raise ValueError("Nonfinite TRAIN optimizer result")
        records.append(dict(parameters=dict(zip(PARAMETER_KEYS, map(float, solve.x))), cost=float(solve.cost),
            success=bool(solve.success), status=int(solve.status), nfev=int(solve.nfev), message=str(solve.message)))
    selected = min(range(len(records)), key=lambda index: records[index]["cost"])
    return dict(version=VERSION, law=law, parameters=records[selected]["parameters"],
        n_train=len(y), n_unique_phi=len(np.unique(phi)), selected_start=selected,
        cost=records[selected]["cost"], starts=records, scope=SCOPE)


def compose(parent, logprediction, admitted, polybot_mask):
    """Full expert [P,P,P,0,0]; predictions align ALL parent rows, not only active."""
    values, mask, pb = np.asarray(parent), np.asarray(admitted), np.asarray(polybot_mask)
    prediction = _vector(logprediction, "logprediction")
    if (values.ndim != 2 or values.shape[1] != 5 or values.dtype.kind != "f" or
            not np.isfinite(values).all() or prediction.shape != (len(values),) or
            mask.shape != prediction.shape or pb.shape != mask.shape or mask.dtype != bool or pb.dtype != bool):
        raise ValueError("Finite floating parent/predictions and aligned Boolean masks required")
    if np.any(mask & pb) or np.any(np.abs(values[:, 0] - values[:, 2] - values[:, 3]) > 1e-10):
        raise ValueError("PB admission or invalid parent decomposition")
    output = values.copy()
    output[mask, :3], output[mask, 3:] = prediction[mask, None], 0.
    return output
