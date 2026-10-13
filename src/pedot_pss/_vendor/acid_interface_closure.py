"""Input-limited acid release closures applied to a FROZEN transport circuit.

The hydration gate is an initial-neat-concentration proximity hypothesis, not
water activity, pH, dissociation, retained film water or measured PSS removal.
No source ID, target field, file, model, or solver is read by this module.
Only fit's explicit y is supervised; callers must supply training-fold base
components from a separately fitted/frozen prior. Query components are never fit.
"""
from __future__ import annotations

from collections import Counter
import math
import numbers
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares


VERSION = "acid-interface-release-closure-v1"
INPUT_VERSION = "typed-applied-post-solution-overlay-v2"
PREFIX = "typed_condition_"
IDENTITY_KEY = PREFIX + "post_solution_acid_identity"
CONCENTRATION_KEY = PREFIX + "post_solution_concentration_mol_L"
HAS_KEY = PREFIX + "has_post_solution_concentration_mol_L"
POSITIVE_KEY = PREFIX + "post_solution_positive_concentration"
MODES = ("control", "release_only", "hydration_release", "replace_binary_release")
GATED_MODES = ("hydration_release", "replace_binary_release")
PROTOCOL_SCOPE = "aqueous-hotdrop-dry-rinse-pilot-v1"
OLD_ACID_FLAG_KEY = "domain_acid_post_present"
# Caller verifies BOTH old preclip x*beta and x*beta+beta_acid in (-30,30).
# Otherwise multiplying the clipped old rho would not exactly cancel its term.
OLD_LINEAR_UNCLIPPED_KEY = "acid_interface_old_log_rho_unclipped"
LN10 = math.log(10.)
# Original N011 initial-solution ranges, not inferred from conductivity targets.
# No extrapolation of this local closure to a new concentration/species domain.
SUPPORTED_MAX_M = {"acetic_acid": 16., "propionic_acid": 13.,
                   "butyric_acid": 11., "oxalic_acid": 1.,
                   "hydrochloric_acid": 12., "sulfurous_acid": .8}
MIN_POSITIVE_M = .1
# N011 (DOI 10.1021/am900708x), pure monoacid endpoints in the original text.
# c/c_neat is ONLY a dimensionless proxy, not mole fraction or water activity.
NEAT_ENDPOINT_M = {"acetic_acid": 16., "propionic_acid": 13., "butyric_acid": 11.}
PARAMETER_BOUNDS = {"release_amplitude": (0., 6.),
                    "half_saturation_mol_L": (.01, 100.),
                    "hydration_exponent": (.1, 8.)}


def _finite_number(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)))


def acid_coordinates(rows):
    """Return explicit input availability/status; missing c is None, not zero.

    Admission relies on the upstream v2 exact applied-solution evidence parser.
    Bare additive names, old post_bath keys and narrative substring matches do
    not activate this closure. An explicit zero is admitted but has no effect.
    """
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("Each input row must be a mapping")
        acid, value = row.get(IDENTITY_KEY), row.get(CONCENTRATION_KEY)
        status = "admitted"
        if not isinstance(acid, str) or acid not in SUPPORTED_MAX_M:
            status = "unsupported_identity"
        elif not _finite_number(row.get(HAS_KEY)) or row[HAS_KEY] != 1.:
            status = "missing_concentration"
        elif not _finite_number(value) or value < 0:
            status = "invalid_concentration"
        elif POSITIVE_KEY in row and (not _finite_number(row[POSITIVE_KEY]) or
                                      row[POSITIVE_KEY] != float(value > 0)):
            status = "inconsistent_positive_flag"
        elif value != 0 and not MIN_POSITIVE_M <= value <= SUPPORTED_MAX_M[acid]:
            status = "outside_supported_concentration"
        admitted = status == "admitted"
        endpoint = NEAT_ENDPOINT_M.get(acid) if isinstance(acid, str) else None
        result.append(dict(status=status, acid_identity=acid if admitted else None,
                           has_concentration=admitted,
                           concentration_mol_L=float(value) if admitted else None,
                           has_neat_proxy=bool(admitted and endpoint is not None),
                           neat_ratio=float(value/endpoint) if admitted and endpoint else None))
    return result


def _parameters(parameters, mode):
    if mode not in MODES:
        raise ValueError("Unknown acid closure mode")
    required = {"release_amplitude", "half_saturation_mol_L"}
    if mode in GATED_MODES:
        required.add("hydration_exponent")
    if not isinstance(parameters, Mapping) or not required.issubset(parameters):
        raise ValueError("Missing explicit acid closure parameters")
    out = {}
    for key in required:
        value = parameters[key]
        low, high = PARAMETER_BOUNDS[key]
        if not _finite_number(value) or not low <= value <= high:
            raise ValueError("Acid parameter outside finite declared bounds: " + key)
        out[key] = float(value)
    return out


def release_coordinates(rows, parameters, *, mode="release_only", protocol_scope=None):
    """Pure input transform. Explicit endpoint branch avoids 0**0/log(0)."""
    pars = _parameters(parameters, mode)
    if mode in GATED_MODES and protocol_scope != PROTOCOL_SCOPE:
        raise ValueError("Neat-endpoint hypothesis requires explicit verified pilot protocol scope")
    rows = list(rows)
    coords = acid_coordinates(rows)
    n = len(coords)
    saturation, gate = np.zeros(n), np.zeros(n)
    active, has_neat, ratios = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, coord in enumerate(coords):
        if not coord["has_concentration"]:
            continue
        c = coord["concentration_mol_L"]
        active[i] = float(c > 0)
        saturation[i] = c / (pars["half_saturation_mol_L"] + c)
        gate[i] = 1.
        if coord["has_neat_proxy"]:
            has_neat[i], ratios[i] = 1., coord["neat_ratio"]
            if mode in GATED_MODES:
                q = coord["neat_ratio"]
                gate[i] = 0. if q == 1. else math.exp(pars["hydration_exponent"] * math.log1p(-q))
    coordinate = saturation * gate
    if mode == "control":
        coordinate = np.zeros(n)
    if mode == "replace_binary_release":
        has_old_acid = np.asarray([float(_finite_number(r.get(OLD_ACID_FLAG_KEY)) and
                                        r[OLD_ACID_FLAG_KEY] == 1. and
                                        _finite_number(r.get(OLD_LINEAR_UNCLIPPED_KEY)) and
                                        r[OLD_LINEAR_UNCLIPPED_KEY] == 1.) for r in rows])
        active *= has_old_acid
        coordinate *= has_old_acid
    return dict(acid_has_applied_solution_concentration=np.asarray(
                    [float(c["has_concentration"]) for c in coords]),
                acid_is_active=active, acid_has_neat_proxy=has_neat,
                acid_neat_concentration_ratio=ratios, acid_saturation=saturation,
                acid_hydration_gate=gate, acid_release_coordinate=coordinate)


def _circuit(gem, rho, bypass):
    log_rho = np.full(len(rho), -np.inf)
    np.log(rho, out=log_rho, where=rho > 0)
    log_denom = np.logaddexp(0., log_rho + gem * LN10)
    log_p, log_not_p = np.full(len(rho), -np.inf), np.full(len(rho), -np.inf)
    np.log(bypass, out=log_p, where=bypass > 0)
    np.log1p(-bypass, out=log_not_p, where=bypass < 1.)
    correction = np.logaddexp(log_p, log_not_p-log_denom) / LN10
    prediction = gem + correction
    sigma_gem, sigma_series, sigma = 10.**gem, 10.**(gem-log_denom/LN10), 10.**prediction
    if not all(np.isfinite(v).all() for v in (prediction, sigma_gem, sigma_series, sigma)) or (sigma <= 0).any():
        raise ValueError("Circuit outside representable positive conductivity range")
    expected = bypass*sigma_gem + (1.-bypass)*sigma_series
    closure = (sigma-expected)/np.maximum(expected, 1e-300)
    return dict(prediction_log10_S_cm=prediction, gem_log10=gem.copy(),
                domain_correction_log10=correction, sigma_domain_S_cm=sigma,
                sigma_gem_S_cm=sigma_gem, sigma_series_S_cm=sigma_series,
                series_correction_log10=-log_denom/LN10,
                series_resistivity_ohm_cm=rho, barrier_rho_ohm_cm=rho,
                bypass_fraction=bypass.copy(), closure_residual=closure,
                domain_bound_violation=((sigma < sigma_series*(1.-1e-11)) |
                                        (sigma > sigma_gem*(1.+1e-11))).astype(float))


def _base_arrays(base_components, n):
    if not isinstance(base_components, Mapping):
        raise TypeError("Frozen base components must be a mapping")
    required = ("gem_log10", "series_resistivity_ohm_cm", "bypass_fraction",
                "prediction_log10_S_cm")
    result = {}
    for key in required:
        value = np.asarray(base_components.get(key), dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all():
            raise ValueError("Missing/nonfinite/misaligned frozen base component: " + key)
        result[key] = value.copy()
    if (result["series_resistivity_ohm_cm"] < 0).any() or (result["bypass_fraction"] < 0).any() or (result["bypass_fraction"] > 1).any():
        raise ValueError("Frozen circuit requires nonnegative rho and 0<=bypass<=1")
    checked = _circuit(result["gem_log10"], result["series_resistivity_ohm_cm"], result["bypass_fraction"])
    if not np.allclose(checked["prediction_log10_S_cm"], result["prediction_log10_S_cm"], atol=1e-10, rtol=0.):
        raise ValueError("Frozen base prediction does not match its actual circuit")
    return result


def apply_acid_interface(rows, base_components, parameters, *, mode="release_only",
                         protocol_scope=None, frozen_acid_relief=None):
    """Modify only frozen interface rho, preserving GEM and bypass exactly."""
    rows = list(rows)
    pars = _parameters(parameters, mode)
    base = _base_arrays(base_components, len(rows))
    coords = release_coordinates(rows, pars, mode=mode, protocol_scope=protocol_scope)
    delta_ln_rho = -pars["release_amplitude"] * coords["acid_release_coordinate"]
    binary_replacement = np.zeros(len(rows))
    if mode == "replace_binary_release":
        if not _finite_number(frozen_acid_relief) or not 0 <= frozen_acid_relief <= 6:
            raise ValueError("P3 requires the explicit unchanged old acid-relief coefficient in [0,6]")
        binary_replacement = float(frozen_acid_relief) * coords["acid_is_active"]
        delta_ln_rho += binary_replacement
    rho = base["series_resistivity_ohm_cm"] * np.exp(delta_ln_rho)
    result = _circuit(base["gem_log10"], rho, base["bypass_fraction"])
    unchanged = delta_ln_rho == 0
    # Preserve the exact frozen prediction when inactive/zero/at neat endpoint.
    result["prediction_log10_S_cm"][unchanged] = base["prediction_log10_S_cm"][unchanged]
    result["domain_correction_log10"] = result["prediction_log10_S_cm"] - base["gem_log10"]
    result.update(coords)
    result["acid_delta_log_resistivity"] = delta_ln_rho
    result["acid_binary_replacement_log_resistivity"] = binary_replacement
    result["acid_correction_log10"] = result["prediction_log10_S_cm"] - base["prediction_log10_S_cm"]
    result["base_prediction_log10_S_cm"] = base["prediction_log10_S_cm"]
    result["base_series_resistivity_ohm_cm"] = base["series_resistivity_ohm_cm"]
    # Copy only known physical checks; never propagate arbitrary target arrays.
    for key in ("gem_equation_residual", "gem_phase_bound_violation", "anisotropy_factor"):
        if key in base_components:
            value = np.asarray(base_components[key], dtype=float)
            if value.shape != (len(rows),) or not np.isfinite(value).all():
                raise ValueError("Invalid base physical check: " + key)
            result[key] = value.copy()
    return result


class AcidInterfaceClosure:
    """Small JSON-serializable fit over frozen TRAINING-fold circuit components."""
    def __init__(self, mode="release_only", *, regularization=.02, max_nfev=100,
                 protocol_scope=None, frozen_acid_relief=None):
        if mode not in MODES:
            raise ValueError("Unsupported closure mode")
        if not _finite_number(regularization) or regularization < 0:
            raise ValueError("regularization must be finite/nonnegative")
        if type(max_nfev) is not int or max_nfev < 1:
            raise ValueError("max_nfev must be a positive int")
        if mode in GATED_MODES and protocol_scope != PROTOCOL_SCOPE:
            raise ValueError("Explicit verified pilot protocol scope is required")
        if mode == "replace_binary_release" and (not _finite_number(frozen_acid_relief) or not 0 <= frozen_acid_relief <= 6):
            raise ValueError("P3 requires the unchanged frozen old acid-relief coefficient")
        self.mode, self.regularization, self.max_nfev = mode, float(regularization), max_nfev
        self.protocol_scope, self.frozen_acid_relief = protocol_scope, frozen_acid_relief

    def fit(self, rows, y, *, base_components, sample_weight=None):
        rows = list(rows)
        target = np.asarray(y, dtype=float)
        if not len(rows) or target.shape != (len(rows),) or not np.isfinite(target).all():
            raise ValueError("Explicit finite training y must match nonempty rows")
        base = _base_arrays(base_components, len(rows))
        weights = np.ones(len(rows)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != target.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Invalid training sample weights")
        initial = dict(release_amplitude=.5, half_saturation_mol_L=1., hydration_exponent=1.)
        coords = release_coordinates(rows, initial, mode=self.mode, protocol_scope=self.protocol_scope)
        useful = (coords["acid_release_coordinate"] > 0) & (base["series_resistivity_ohm_cm"] > 0) & (base["bypass_fraction"] < 1) & (weights > 0)
        self.n_training_examples_ = len(rows)
        self.n_supported_training_examples_ = int(coords["acid_has_applied_solution_concentration"].sum())
        self.n_effective_training_examples_ = int(useful.sum())
        self.input_status_counts_ = dict(Counter(c["status"] for c in acid_coordinates(rows)))
        self.parameters_ = dict(initial, release_amplitude=0.)
        if self.mode == "control" or not useful.any():
            self.optimization_ = dict(success=True, evaluations=0, status="no_effective_fit_rows")
        else:
            indices = np.flatnonzero(useful)
            selected_rows = [rows[i] for i in indices]
            selected_base = {key: value[indices] for key, value in base.items()}
            root_w = np.sqrt(weights[indices]/weights[indices].mean())
            lower = np.array([0., math.log(.01)] + ([math.log(.1)] if self.mode in GATED_MODES else []))
            upper = np.array([6., math.log(100.)] + ([math.log(8.)] if self.mode in GATED_MODES else []))
            start = np.array([.5, 0.] + ([0.] if self.mode in GATED_MODES else []))
            # Start away from the bound for numerical sensitivity, but shrink
            # the NEW mechanism toward its null amplitude, not toward A=.5.
            penalty_center = np.zeros_like(start)

            def unpack(values):
                return dict(release_amplitude=float(values[0]),
                            half_saturation_mol_L=min(100., max(.01, math.exp(float(values[1])))),
                            hydration_exponent=min(8., max(.1, math.exp(float(values[2])))) if len(values) == 3 else 1.)

            def objective(values):
                predicted = apply_acid_interface(selected_rows, selected_base, unpack(values), mode=self.mode,
                    protocol_scope=self.protocol_scope, frozen_acid_relief=self.frozen_acid_relief)["prediction_log10_S_cm"]
                penalty = math.sqrt(len(indices)*self.regularization)*(values-penalty_center)
                return np.r_[root_w*(predicted-target[indices]), penalty]

            solution = least_squares(objective, start, bounds=(lower, upper), max_nfev=self.max_nfev,
                                     ftol=1e-9, xtol=1e-9, gtol=1e-9)
            self.parameters_ = unpack(solution.x)
            self.optimization_ = dict(success=bool(solution.success), evaluations=int(solution.nfev),
                                      status=int(solution.status), message=str(solution.message))
        parts = self.components(rows, base_components=base)
        self.training_checks_ = dict(circuit_residual_max_abs=float(np.max(abs(parts["closure_residual"]))),
                                    bound_violations=int(parts["domain_bound_violation"].sum()),
                                    added_correction_max_abs_log10=float(np.max(abs(parts["acid_correction_log10"]))),
                                    training_rmse_log10=float(np.sqrt(np.mean((parts["prediction_log10_S_cm"]-target)**2))))
        return self

    def components(self, rows, *, base_components):
        if not hasattr(self, "parameters_"):
            raise RuntimeError("AcidInterfaceClosure must be fit on training components first")
        return apply_acid_interface(rows, base_components, self.parameters_, mode=self.mode,
                                   protocol_scope=self.protocol_scope, frozen_acid_relief=self.frozen_acid_relief)

    def predict(self, rows, *, base_components):
        return self.components(rows, base_components=base_components)["prediction_log10_S_cm"]

    def summary(self):
        if not hasattr(self, "parameters_"):
            raise RuntimeError("Closure is not fit")
        return dict(version=VERSION, input_version=INPUT_VERSION, mode=self.mode,
                    protocol_scope=self.protocol_scope, frozen_acid_relief=self.frozen_acid_relief,
                    parameters=dict(self.parameters_), bounds=dict(PARAMETER_BOUNDS),
                    n_training_examples=self.n_training_examples_,
                    n_supported_training_examples=self.n_supported_training_examples_,
                    n_effective_training_examples=self.n_effective_training_examples_,
                    input_status_counts=self.input_status_counts_, optimization=self.optimization_,
                    checks=self.training_checks_, regularization=self.regularization,
                    regularization_center=dict(release_amplitude=0., log_half_saturation=0., log_hydration_exponent=0.),
                    max_nfev=self.max_nfev, neat_endpoint_mol_L=dict(NEAT_ENDPOINT_M),
                    supported_positive_range_mol_L={k: [MIN_POSITIVE_M, v] for k, v in SUPPORTED_MAX_M.items()},
                    units={"rho": "ohm cm", "c": "mol/L", "amplitude": "dimensionless natural-log rho", "correction": "log10 conductivity difference"},
                    limits=["Effective shared release parameters; not measured PSS loss, pH, water activity, wet-film temperature, oxidation or microscopic kinetics.",
                            "Initial neat-concentration ratio applies only to three named monoacids; no extrapolation beyond declared input domain.",
                            "Pure endpoint has zero NEW release, not untreated conductivity: frozen base may already contain an acid-binary effect.",
                            "P3 explicitly cancels the old acid binary relief only on supported positive rows with old flag=1 and explicit old-log-rho-unclipped=1; old coefficient is supplied frozen, never fit.",
                            "The gate is initial-neat-endpoint proximity, not actual hydration; caller must bind the declared aqueous hotdrop/dry/rinse protocol. Shared K/beta need not explain acid-specific peak locations.",
                            "Frozen base GEM, bypass and all old coefficients remain unchanged; caller must ensure train-only prior fitting and equal residual information/capacity.",
                            "No future wet-film path is inferred from query conductivity; single-paper CV is not cross-paper mechanism validation."])
