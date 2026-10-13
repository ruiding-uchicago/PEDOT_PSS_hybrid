"""Actual GEM plus positive PSS-interface resistor-network closures.

This is a grey-box effective transport model, not a PINN or microscopic
parameter identification. Process coordinates are input-only; fitting uses
only the targets explicitly supplied to fit. Units: sigma S/cm, rho ohm*cm.
The additional circuit changes the physical prediction before any ML residual.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

try:
    from .calibrated_hybrid import _prepare_rows
    from .hybrid import _physics
except ImportError:
    from .calibrated_hybrid import _prepare_rows
    from .hybrid import _physics


MODES = ("gem_only", "interface_series", "anisotropic_series",
         "parallel_bypass", "anisotropic_bypass")
PARAMETER_NAMES = (
    "log_rho_reference_ohm_cm", "pss_barrier_strength", "dmso_barrier_relief",
    "eg_barrier_relief", "post_polar_barrier_relief", "acid_barrier_relief",
    "thermal_barrier_response", "out_of_plane_log_resistivity_factor",
    "bypass_logit", "bypass_solvent_response",
)
LN10 = math.log(10.)


def _number(row, *keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, (numbers.Real, np.number)) and not isinstance(value, (bool, np.bool_)):
            if math.isfinite(float(value)):
                return float(value)
    return None


def _coordinates(prepared, mechanism_rows, design):
    result = []
    for row, mechanism, physical in zip(prepared, mechanism_rows, design):
        ratio = _number(row, "reported_pss_pedot_mass_ratio")
        dmso = _number(row, "dmso_dose_volpct", "dmso_volpct")
        eg = _number(row, "eg_dose_volpct", "eg_volpct")
        coverage = _number(row, "structure_coverage_percent")
        # A reported solution mass ratio is a process covariate, NEVER a
        # dry-film volume fraction, PSS shell thickness, or measured barrier.
        valid_ratio = ratio is not None and ratio >= 0.
        post_values = [mechanism.get(f"coord_post_{chemical}_saturation_volpct", 0.)
                       for chemical in ("dmso", "eg", "methanol", "ethanol", "ipa", "dmf")]
        post_present = float(row.get("polar_solvent_post_present", 0.) or 0.)
        post = max(min(sum(post_values), 1.), min(max(post_present, 0.), 1.))
        has_direction = float("measurement_out_of_plane" in row or
                              bool(str(row.get("measurement_direction", "")).strip()))
        out = {
            "domain_has_pss_mass_ratio": float(valid_ratio),
            "domain_pss_mass_ratio_log1p": math.log1p(min(ratio, 100.)) / math.log(31.) if valid_ratio else 0.,
            "domain_has_dmso_volpct": float(dmso is not None and dmso >= 0.),
            "domain_dmso_saturation": float(physical[1]),
            "domain_has_eg_volpct": float(eg is not None and eg >= 0.),
            "domain_eg_saturation": float(physical[2]),
            "domain_post_polar_saturation": float(post),
            "domain_acid_post_present": float(physical[3]),
            "domain_has_thermal_budget": float(mechanism["coord_has_total_thermal_budget"]),
            "domain_thermal_saturation": float(mechanism["coord_total_thermal_saturation"]),
            "domain_has_direction_flag": has_direction,
            "domain_out_of_plane": float(physical[-1]),
            # Observed first-dry coverage is exposed for a residual model only.
            # It is intentionally absent from the resistor-network closure.
            "domain_has_observed_coverage": float(coverage is not None and 0. <= coverage <= 100.),
            "domain_observed_coverage_fraction": coverage / 100. if coverage is not None and 0. <= coverage <= 100. else 0.,
        }
        result.append(out)
    return result


def _inputs(rows, activation_energy_eV):
    prepared, mechanisms = _prepare_rows(rows, activation_energy_eV, "coordinates")
    design = _physics.process_design(prepared)
    coordinates = _coordinates(prepared, mechanisms, design)
    matrix = np.asarray([[1., r["domain_pss_mass_ratio_log1p"],
                          -r["domain_dmso_saturation"], -r["domain_eg_saturation"],
                          -r["domain_post_polar_saturation"], -r["domain_acid_post_present"],
                          r["domain_thermal_saturation"], r["domain_out_of_plane"],
                          .5 * (r["domain_dmso_saturation"] + r["domain_eg_saturation"])
                          + r["domain_post_polar_saturation"]] for r in coordinates], dtype=float)
    return prepared, design, coordinates, matrix.reshape(len(coordinates), 9)


def domain_coordinates(rows, activation_energy_eV=.35):
    """Return fresh input-only numeric dictionaries, including missingness.

    Takes parsed reconstructed_state/enriched feature dictionaries, not raw
    original_row containers. Existing input validation rejects targets/IDs.
    No fit, measured electrical target, assay SD, or fitted inverse is read.
    """
    return _inputs(rows, activation_energy_eV)[2]


def resistor_network(gem_log10, coordinate_matrix, parameters, mode, interface_scale=1.):
    """Positive series resistance and bounded parallel paths; no fitted state.

    rho_b = scale * exp(x beta) [ohm cm]; sigma_s = 1/(1/sigma_G + rho_b).
    Optional anisotropy multiplies rho_b for an explicit out-of-plane flag.
    Optional bypass: sigma = p*sigma_G + (1-p)*sigma_s, 0<p<1. p is an
    effective bypass-path weight, NOT observed coverage or a volume fraction.
    """
    if mode not in MODES: raise ValueError("Unknown domain transport mode")
    if not math.isfinite(interface_scale) or interface_scale < 0:
        raise ValueError("interface_scale must be finite and nonnegative")
    gem = np.asarray(gem_log10, dtype=float)
    x = np.asarray(coordinate_matrix, dtype=float)
    beta = np.asarray(parameters, dtype=float)
    if gem.ndim != 1 or x.shape != (len(gem), 9) or beta.shape != (10,):
        raise ValueError("Invalid circuit dimensions")
    if not np.isfinite(gem).all() or not np.isfinite(x).all() or not np.isfinite(beta).all():
        raise ValueError("Circuit inputs must be finite")
    anisotropic = mode in {"anisotropic_series", "anisotropic_bypass"}
    bypass = mode in {"parallel_bypass", "anisotropic_bypass"}
    anisotropy = np.exp(np.clip(beta[7] * x[:, 7], -30., 30.)) if anisotropic else np.ones(len(gem))
    rho = interface_scale * np.exp(np.clip(x[:, :7] @ beta[:7], -30., 30.)) * anisotropy
    if mode == "gem_only": rho = np.zeros(len(gem))
    sigma_gem = 10. ** gem
    loading = rho * sigma_gem
    series_correction = -np.log1p(loading) / LN10
    sigma_series = sigma_gem / (1. + loading)
    fraction = expit(beta[8] + beta[9] * x[:, 8]) if bypass else np.zeros(len(gem))
    if mode == "gem_only": fraction = np.zeros(len(gem))
    # Stable even when rho*sigma is tiny. The domain correction remains
    # explicitly visible instead of being lost by subtracting two log values.
    if bypass:
        correction = np.logaddexp(np.log(fraction),
                     np.log1p(-fraction) - np.log1p(loading)) / LN10
    else:
        correction = series_correction
    prediction = gem + correction
    sigma = 10. ** prediction
    expected = fraction * sigma_gem + (1. - fraction) * sigma_series
    closure_residual = (sigma - expected) / np.maximum(expected, 1e-300)
    return dict(prediction_log10_S_cm=prediction, gem_log10=gem,
                domain_correction_log10=correction, series_correction_log10=series_correction,
                sigma_domain_S_cm=sigma, sigma_gem_S_cm=sigma_gem,
                sigma_series_S_cm=sigma_series, series_resistivity_ohm_cm=rho,
                barrier_rho_ohm_cm=rho, anisotropy_factor=anisotropy,
                bypass_fraction=fraction, closure_residual=closure_residual,
                domain_bound_violation=((sigma < sigma_series * (1. - 1e-11)) |
                                        (sigma > sigma_gem * (1. + 1e-11))).astype(float))


class DomainTransport(RegressorMixin, BaseEstimator):
    """Shared fold-local actual GEM + effective interfacial circuit closure."""
    def __init__(self, mode="interface_series", coupling="sequential", max_nfev=150,
                 random_state=20261009, regularization=.03, gem_regularization=.01,
                 activation_energy_eV=.35, interface_scale=1.):
        self.mode, self.coupling, self.max_nfev = mode, coupling, max_nfev
        self.random_state, self.regularization = random_state, regularization
        self.gem_regularization = gem_regularization
        self.activation_energy_eV, self.interface_scale = activation_energy_eV, interface_scale

    def fit(self, rows, y, sample_weight=None):
        rows = list(rows)
        if self.mode not in MODES or self.coupling not in {"sequential", "joint"}:
            raise ValueError("Unsupported mode/coupling")
        if self.max_nfev < 1 or min(self.regularization, self.gem_regularization) < 0:
            raise ValueError("Invalid iteration or regularization parameters")
        if not math.isfinite(self.interface_scale) or self.interface_scale < 0:
            raise ValueError("interface_scale must be finite and nonnegative")
        if not math.isfinite(self.activation_energy_eV) or self.activation_energy_eV < 0:
            raise ValueError("activation_energy_eV must be finite and nonnegative")
        prepared, design, _, matrix = _inputs(rows, self.activation_energy_eV)
        target = np.asarray(y, dtype=float)
        if target.shape != (len(prepared),) or not len(target) or not np.isfinite(target).all():
            raise ValueError("Finite nonempty log10 conductivity targets must match rows")
        weights = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight, float)
        if weights.shape != target.shape or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Invalid sample weights")
        weights = weights / weights.mean()
        physics = _physics.GreyBoxTransport(max_nfev=self.max_nfev, enable_hopping=False,
                     random_state=self.random_state, regularization=self.gem_regularization)
        physics.fit(prepared, target, sample_weight=weights)
        base_coef = physics.coef_.copy()
        base_gem = physics.predict(prepared)
        prior = np.asarray([np.clip(-np.median(base_gem) * LN10 + math.log(.2), -15., 15.),
                            .25, .4, .4, .2, .2, 0., math.log(3.), -math.log(4.), .5])
        lower = np.asarray([-18., 0., 0., 0., 0., 0., -4., 0., -6., 0.])
        upper = np.asarray([18., 6., 6., 6., 6., 6., 4., math.log(1e4), 6., 6.])
        active = list(range(7))
        if self.mode in {"anisotropic_series", "anisotropic_bypass"}: active += [7]
        if self.mode in {"parallel_bypass", "anisotropic_bypass"}: active += [8, 9]
        if self.mode == "gem_only" or self.interface_scale == 0.: active = []
        joint = self.coupling == "joint" and bool(active)
        n_gem = len(_physics.FEATURE_NAMES)
        gem_lower, gem_upper = np.full(n_gem, -4.), np.full(n_gem, 4.)
        gem_lower[0], gem_upper[0] = -7., 7.
        gem_lower[1:3] = 0.; gem_upper[5] = 0.
        start = np.r_[base_coef[:n_gem], prior[active]] if joint else prior[active]
        lo = np.r_[gem_lower, lower[active]] if joint else lower[active]
        hi = np.r_[gem_upper, upper[active]] if joint else upper[active]

        def unpack(values):
            coef, beta = base_coef.copy(), prior.copy()
            if joint: coef[:n_gem], values = values[:n_gem], values[n_gem:]
            beta[active] = values
            return coef, beta

        def objective(values):
            coef, beta = unpack(values)
            gem = physics._compute(design, np.zeros(len(target)), coef)[0] if joint else base_gem
            predicted = resistor_network(gem, matrix, beta, self.mode, self.interface_scale)["prediction_log10_S_cm"]
            penalty_scale = np.ones(len(active))
            if active: penalty_scale[0] = .25
            penalty = math.sqrt(len(target) * self.regularization) * (beta[active] - prior[active]) * penalty_scale
            if joint:
                gem_penalty = math.sqrt(len(target) * self.gem_regularization) * (coef[:n_gem] - base_coef[:n_gem])
                penalty = np.r_[penalty, gem_penalty]
            return np.r_[np.sqrt(weights) * (predicted - target), penalty]

        if active:
            solution = least_squares(objective, np.clip(start, lo+1e-9, hi-1e-9), bounds=(lo, hi),
                                     max_nfev=self.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            physics.coef_, beta = unpack(solution.x)
            optimization = dict(success=bool(solution.success), evaluations=int(solution.nfev),
                                status=int(solution.status), message=str(solution.message))
        else:
            beta = prior.copy()
            optimization = dict(success=True, evaluations=0, status=0, message="GEM-only ablation; no domain optimization")
        self.gem_, self.domain_parameters_ = physics, beta
        self.n_training_examples_ = len(target)
        self.active_domain_indices_, self.optimization_ = active, optimization
        train_parts = self.components(rows)
        self.training_checks_ = dict(
            gem_equation_residual_max_abs=float(np.max(np.abs(train_parts["gem_equation_residual"]))),
            circuit_closure_residual_max_abs=float(np.max(np.abs(train_parts["closure_residual"]))),
            domain_correction_max_abs_log10=float(np.max(np.abs(train_parts["domain_correction_log10"]))),
            domain_correction_median_abs_log10=float(np.median(np.abs(train_parts["domain_correction_log10"]))),
            domain_bound_violations=int(np.sum(train_parts["domain_bound_violation"])),
            training_rmse_log10=float(np.sqrt(np.mean((train_parts["prediction_log10_S_cm"] - target)**2))))
        return self

    def components(self, rows):
        check_is_fitted(self, ["gem_", "domain_parameters_"])
        prepared, _, _, matrix = _inputs(rows, self.activation_energy_eV)
        if not prepared:
            return {key: np.empty(0) for key in ["prediction_log10_S_cm", "gem_log10", "domain_correction_log10",
                "series_correction_log10", "sigma_domain_S_cm", "sigma_gem_S_cm", "sigma_series_S_cm",
                "series_resistivity_ohm_cm", "barrier_rho_ohm_cm", "anisotropy_factor", "bypass_fraction",
                "closure_residual", "domain_bound_violation", "gem_equation_residual", "gem_phase_bound_violation"]}
        base = self.gem_.components(prepared)
        result = resistor_network(base["prediction_log10_S_cm"], matrix, self.domain_parameters_, self.mode, self.interface_scale)
        result["gem_equation_residual"] = base["gem_equation_residual"]
        low = base["effective_low_phase_S_cm"]
        result["gem_phase_bound_violation"] = ((result["sigma_gem_S_cm"] < low * (1.-1e-11)) |
            (result["sigma_gem_S_cm"] > self.gem_.sigma_high * (1.+1e-11))).astype(float)
        return result

    physics_components = components

    def predict(self, rows):
        return self.components(rows)["prediction_log10_S_cm"]

    def summary(self):
        check_is_fitted(self, ["gem_", "domain_parameters_"])
        return dict(model="actual GEM plus effective PSS-interface resistor network", mode=self.mode,
                    coupling=self.coupling, n_training_examples=self.n_training_examples_,
                    domain_parameters=dict(zip(PARAMETER_NAMES, self.domain_parameters_.tolist())),
                    fitted_domain_parameters=[PARAMETER_NAMES[i] for i in self.active_domain_indices_],
                    gem=self.gem_.summary(), optimization=self.optimization_, checks=self.training_checks_,
                    interface_scale=self.interface_scale, activation_energy_eV=self.activation_energy_eV,
                    regularization=self.regularization, gem_regularization=self.gem_regularization,
                    prediction_equation="sigma_series=1/(1/sigma_GEM+rho_b); sigma=p*sigma_GEM+(1-p)*sigma_series",
                    units={"conductivity":"S/cm", "resistivity":"ohm cm", "domain_correction":"log10(S/cm) difference"},
                    limits=["Shared effective coefficients, not measured PSS thickness, tunnel length, phase fraction or microscopic identification.",
                            "Only explicitly reported PSS/PEDOT MASS ratio is used; no molar/unresolved-ratio conversion.",
                            "Observed first-dry coverage is not used as a conductive phase volume fraction or in the physical circuit.",
                            "Processing Arrhenius budget uses a fixed .35 eV prior, not a fitted transport activation energy; no universal VRH is assumed.",
                            "Solvent relief signs are a bounded hypothesis for this pooled closure, not a universal guarantee for every chemistry.",
                            "GEM phase bounds apply before the added interface; the final circuit can lie below the low GEM phase.",
                            "Small circuit residual verifies numerical equations, not mechanism validity or physical dominance."])


DomainTransportV2 = DomainTransport
