"""Two reviewed outer-polymer hosts; fixed nominal densities, not occupancy.

Only caller-supplied TRAIN responses enter fit. Family1 is the reviewed
synthetic high-Mw PEDOT:PSS/DMSO/PEO8e6 formulation, not a source-ID feature.
The morphology term is a phenomenological connectivity-loss prior, not an
identified microscopic mechanism. No IO, query inverse, free density or mutation
of accepted old-host parameters; TRANSFER preserves their forward exactly.
Caller authenticates scientific specimen joins.
"""
import math
import numbers
from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares

from . import peo_composite_transport as old
VERSION = "reviewed-peo-cross-host-v1"
ARMS = ("TRANSFER", "SHARED_PERC", "SHARED_MORPH", "MATCHED_LOGPHI")
MU_BOUNDS = (-6., math.log10(2e4))
DENSITIES = (1.55, 1.12)
MORPH_ONSET, MORPH_SCALE = .37, .63
REVIEWED_MASSES = (0., .20, .37, .40, .50, .65, .80)
PARAMETER_KEYS = {
    "TRANSFER": ("mu1",), "SHARED_PERC": ("mu0", "mu1", "fc", "t"),
    "SHARED_MORPH": ("mu0", "mu1", "t", "k"),
    "MATCHED_LOGPHI": ("mu0", "mu1", "c", "d")}
BOUNDS = {
    "TRANSFER": ((MU_BOUNDS[0],), (MU_BOUNDS[1],)),
    "SHARED_PERC": ((-6., -6., .005, .3), (MU_BOUNDS[1], MU_BOUNDS[1], .25, 3.)),
    "SHARED_MORPH": ((-6., -6., .3, 0.), (MU_BOUNDS[1], MU_BOUNDS[1], 3., 8.)),
    "MATCHED_LOGPHI": ((-6., -6., -8., -8.), (MU_BOUNDS[1], MU_BOUNDS[1], 8., 8.))}
STARTS = {
    "TRANSFER": ((2.,), (3.,), (4.,)),
    "SHARED_PERC": ((2., 2., .03, 1.), (3., 3., .06, 1.), (4., 4., .12, 2.)),
    "SHARED_MORPH": ((2., 2., 1., 1.), (3., 3., 1., 2.), (4., 4., 2., 4.)),
    "MATCHED_LOGPHI": ((2., 2., 1., 0.), (3., 3., 1., 1.), (4., 4., 2., -1.))}
SCOPE = ("Reviewed outer-polymer nominal dry-mass/density coordinate; density "
         "priors are not measured retained volume or microscopic occupancy. "
         "Posthoc cross-host development, not nested/prospective validation.")

# Controlled raw-input literals; no sample, series, IDs, locators or outcomes.
RAW_PROFILE = {
    "polymer_grade": "high-Mw PEDOT:PSS (PSS Mw=1000 kg mol-1); PEO8000 (Mw=8000 kg mol-1)",
    "stock_solids_scope": "synthesized high-Mw PEDOT:PSS dispersion",
    "stock_solids_wt_percent": .88,
    "pedot_pss_ratio_raw": "1:2.5 (PEDOT:PSS weight ratio)",
    "dose_basis": "PEO wt% = 100 x weight(PEO)/[weight(PEO)+weight(PEDOT:PSS)]; DMSO concentration in the aqueous PEDOT:PSS/PEO solution",
    "specimen_geometry": "free-standing film cut to dog-bone shape, 2 x 30 mm2",
    "substrate": "glass for drop-cast film; free-standing film for conductivity testing",
    "deposition_method": "drop-cast; free-standing PTFE-molded film also prepared for tests",
    "process_sequence": "PEDOT:PSS dispersion sonicated 30 min; concentrated to 2.5 wt%; blended with 2.5 wt% PEO solution and stirred at 50 C overnight; DMSO added; drop-cast or poured into PTFE mold; dried at 70 C (free-standing film overnight)",
    "measurement_method": "four-probe in-line contact method with Keithley 2450 source meter",
    "dry_temperature_C": 70., "measurement_temperature_C": 25.,
    "dry_duration_min": None, "anneal_temperature_C": None,
    "anneal_duration_min": None, "post_treatment": None,
    "post_treatment_temperature_C": None, "post_treatment_duration_min": None,
    "measurement_direction": None, "measurement_quantity": None}
INPUT_FIELDS = tuple(RAW_PROFILE) + ("additive_names", "additive_amounts_raw")


def configuration():
    """Fresh JSON-safe scientific configuration, with no fitted/input state."""
    return dict(version=VERSION, arms=list(ARMS),
        parameter_keys={a: list(PARAMETER_KEYS[a]) for a in ARMS},
        parameter_bounds={a: [list(v) for v in BOUNDS[a]] for a in ARMS},
        starts={a: [list(v) for v in STARTS[a]] for a in ARMS},
        nominal_density_priors_g_cm3=list(DENSITIES),
        structural_onset_peo_mass_fraction=MORPH_ONSET, morphology_scale=MORPH_SCALE,
        percolation_floor_S_cm=old.SIGMA_LOW_S_CM, maximum_nfev_per_start=300,
        TRAIN_only=True, free_density=False, scope=SCOPE)


def _number(value):
    if not isinstance(value, numbers.Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def condition_new(original_row, expected_peo_mass_fraction):
    """Caller-reviewed specimen join plus closed input facts; unknown is inactive."""
    result = dict(active=False, phi_nominal=None, family=1, peo_mass_fraction=None,
                  host_mass_fraction=None, facts=None, reason="unknown_or_conflicting_input", scope=SCOPE)
    if (not isinstance(original_row, Mapping) or not _number(expected_peo_mass_fraction)
            or float(expected_peo_mass_fraction) not in REVIEWED_MASSES):
        return result
    for key, expected in RAW_PROFILE.items():
        actual = original_row.get(key)
        if isinstance(expected, float):
            if not _number(actual) or float(actual) != expected:
                return result
        elif type(actual) is not type(expected) or actual != expected:
            return result
    mass = float(expected_peo_mass_fraction)
    names, amounts = original_row.get("additive_names"), original_row.get("additive_amounts_raw")
    if (type(names) is not list or type(amounts) is not list or
            any(type(v) is not str for v in names + amounts) or
            names != ["PEO8000", "DMSO"] or amounts != [
                f"{mass * 100:g} wt% of PEO/(PEO+PEDOT:PSS) blend", "2 wt% in aqueous solution"]):
        return result
    w = 1. - mass
    phi = (w / DENSITIES[0]) / (w / DENSITIES[0] + mass / DENSITIES[1])
    facts = dict(host_identity="synthetic_high_Mw_PEDOT_PSS_DMSO", PSS_Mw_g_mol=1e6,
        internal_PEDOT_PSS_mass_ratio=[1., 2.5], PEO_Mw_g_mol=8e6,
        DMSO_reported_wt_percent=2., DMSO_exact_mass_denominator=None,
        nominal_density_priors_g_cm3=list(DENSITIES), measured_volume_fraction=False,
        dry_duration_s=None, dry_duration_text="overnight", retained_water_DMSO=None,
        reviewed_initial_comparison_temperature_C=25., reviewed_ambient_RH_percent=60.)
    return dict(result, active=True, phi_nominal=float(phi), peo_mass_fraction=mass,
                host_mass_fraction=w, facts=facts, reason="reviewed_new_host_protocol")


def _vector(value, name, integer=False):
    raw = np.asarray(value)
    allowed = "iuf" if integer and raw.size == 0 else ("iu" if integer else "iuf")
    if raw.ndim != 1 or raw.dtype.kind not in allowed:
        raise ValueError(name + " must be a numeric non-Boolean 1D vector")
    if not isinstance(value, np.ndarray) and any(isinstance(v, (bool, np.bool_))
                                               for v in np.asarray(value, dtype=object)):
        raise ValueError(name + " cannot contain Boolean values")
    result = np.asarray(raw, dtype=int if integer else float)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite")
    return result


def _inputs(phi, family, peo_mass):
    phi, family, mass = _vector(phi, "phi"), _vector(family, "family", True), _vector(peo_mass, "peo_mass")
    if (phi.shape != family.shape or phi.shape != mass.shape or np.any((phi <= 0) | (phi > 1))
            or np.any((family != 0) & (family != 1)) or np.any((mass < 0) | (mass >= 1))):
        raise ValueError("Aligned family0/1, phi(0,1] and PEO mass[0,1) required")
    w = 1. - mass
    expected = (w / DENSITIES[0]) / (w / DENSITIES[0] + mass / DENSITIES[1])
    if np.any(abs(phi - expected) > 1e-12):
        raise ValueError("Fixed nominal mass/density conversion differs")
    return phi, family, mass


def _parameters(arm, parameters, transfer_parent):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Unknown arm")
    if not isinstance(parameters, Mapping) or set(parameters) != set(PARAMETER_KEYS[arm]):
        raise ValueError("Exact arm parameter keys required; no free density")
    values = tuple(parameters[k] for k in PARAMETER_KEYS[arm])
    if any(not _number(v) or not lo <= v <= hi for v, lo, hi in zip(values, *BOUNDS[arm])):
        raise ValueError("Finite non-Boolean bounded parameters required")
    if arm == "TRANSFER":
        if not isinstance(transfer_parent, Mapping) or set(transfer_parent) != set(old.PARAMETER_KEYS):
            raise ValueError("Saved old-host parameters required")
        old.forward(np.empty(0), law="percolation", **transfer_parent)
    elif transfer_parent is not None:
        raise ValueError("Only TRANSFER accepts old-host parameters")


def forward(phi, family, peo_mass, arm, parameters, transfer_parent=None):
    """No labels: actual positive conductivity, with empty-query support."""
    phi, family, mass = _inputs(phi, family, peo_mass)
    _parameters(arm, parameters, transfer_parent)
    morphology = np.where(family == 1, np.maximum(0., mass - MORPH_ONSET) / MORPH_SCALE, 0.)
    mu = np.where(family == 0, parameters.get("mu0", 0.), parameters["mu1"])
    if arm in ("TRANSFER", "SHARED_PERC"):
        if arm == "TRANSFER":
            mu = np.where(family == 0, transfer_parent["log10_sigma_host"], parameters["mu1"])
            sigma = old.forward(phi, law="percolation", **transfer_parent)["conductivity_S_cm"].copy()
            new = family == 1
            sigma[new] = old.forward(phi[new], law="percolation", log10_sigma_host=parameters["mu1"],
                                    fc=transfer_parent["fc"], t=transfer_parent["t"])["conductivity_S_cm"]
        else:
            sigma = np.empty_like(phi)
            for h in (0, 1):
                mask = family == h
                sigma[mask] = old.forward(phi[mask], law="percolation", log10_sigma_host=parameters[f"mu{h}"],
                                         fc=parameters["fc"], t=parameters["t"])["conductivity_S_cm"]
        logprediction = np.log10(sigma)
    else:
        logphi = np.log10(phi)
        logprediction = (mu + parameters["t"] * logphi - parameters["k"] * morphology
                         if arm == "SHARED_MORPH" else mu + parameters["c"] * logphi + parameters["d"] * logphi ** 2)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            sigma = np.power(10., logprediction)
    if not np.isfinite(logprediction).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("Unrepresentable positive conductivity")
    diagnostics = dict(arm=arm, scope=SCOPE, empirical_control=arm == "MATCHED_LOGPHI",
                       nominal_phi=phi.copy(), family=family.copy(), peo_mass_fraction=mass.copy())
    if arm != "MATCHED_LOGPHI":
        host = np.power(10., mu)
        lower = old.SIGMA_LOW_S_CM if arm != "SHARED_MORPH" else 0.
        violation = np.maximum(np.maximum(sigma - host, lower - sigma), 0.)
        if np.any(violation > 1e-10):
            raise ValueError("Physical positive phase bounds violated")
        diagnostics.update(host_log10_S_cm=mu, morphology_coordinate=morphology,
            lower_bound_S_cm=lower, phase_bound_violation=violation,
            floor_applied=(sigma == old.SIGMA_LOW_S_CM) if arm != "SHARED_MORPH" else np.zeros(len(phi), bool))
    return dict(prediction_log10_S_cm=logprediction, conductivity_S_cm=sigma, diagnostics=diagnostics)


def _diagnostics(jacobian, values, bounds):
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = np.finfo(float).eps * max(jacobian.shape) * singular[0] if len(singular) else 0.
    rank = int(np.sum(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if len(singular) and rank == jacobian.shape[1] else None
    flags = {k: dict(lower=bool(abs(v - lo) <= 1e-7 * (1 + abs(lo))),
                     upper=bool(abs(v - hi) <= 1e-7 * (1 + abs(hi))))
             for k, v, lo, hi in zip(values, values.values(), *bounds)}
    return dict(jacobian_rank=rank, jacobian_condition_number=condition, bound_flags=flags)


def fit(phi_train, family_train, peo_mass_train, y_train, arm, transfer_parent=None):
    """Three fixed TRAIN-only least-squares starts, max300; no query interface."""
    phi, family, mass = _inputs(phi_train, family_train, peo_mass_train)
    y = _vector(y_train, "y_train")
    if y.shape != phi.shape or type(arm) is not str or arm not in ARMS:
        raise ValueError("Aligned TRAIN labels and supported arm required")
    counts = [int(np.sum(family == h)) for h in (0, 1)]
    distinct = [len(np.unique(phi[family == h])) for h in (0, 1)]
    if ((arm == "TRANSFER" and distinct[1] < 2) or
            (arm != "TRANSFER" and (len(y) < 8 or min(distinct) < 2))):
        raise ValueError("Insufficient TRAIN host/fraction support")
    selected_rows = family == 1 if arm == "TRANSFER" else np.ones(len(y), bool)
    scope = "family1_TRAIN_only" if arm == "TRANSFER" else "all_admitted_TRAIN"
    keys, records = PARAMETER_KEYS[arm], []
    for start in STARTS[arm]:
        def residual(values):
            pars = dict(zip(keys, values))
            return forward(phi, family, mass, arm, pars, transfer_parent)["prediction_log10_S_cm"][selected_rows] - y[selected_rows]
        solved = least_squares(residual, start, bounds=BOUNDS[arm], max_nfev=300)
        pars = dict(zip(keys, map(float, solved.x)))
        actual_cost = .5 * float(np.sum(residual(solved.x) ** 2))
        if (not math.isfinite(actual_cost) or not np.isfinite(solved.jac).all()
                or abs(actual_cost - solved.cost) > 1e-9):
            raise ValueError("Nonfinite or inconsistent TRAIN optimizer result")
        records.append(dict(parameters=pars, cost=actual_cost, success=bool(solved.success),
            status=int(solved.status), nfev=int(solved.nfev), message=str(solved.message),
            **_diagnostics(solved.jac, pars, BOUNDS[arm])))
    selected = min(range(3), key=lambda i: records[i]["cost"])
    diagnostics = {k: records[selected][k] for k in ("jacobian_rank", "jacobian_condition_number", "bound_flags")}
    diagnostics["method"] = "three_fixed_least_squares_starts"
    return dict(version=VERSION, arm=arm, parameters=records[selected]["parameters"],
        n_train=len(y), n_train_by_family=counts, n_unique_phi_by_family=distinct,
        n_families=sum(n > 0 for n in counts), selected_start=selected,
        cost=records[selected]["cost"], cost_scope=scope, starts=records, diagnostics=diagnostics, scope=SCOPE)
