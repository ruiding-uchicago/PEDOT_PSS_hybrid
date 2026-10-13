"""Sparse process closure coupled to an actual generalized-effective-medium solve.

All conductivities are S/cm. Model parameters are effective, not microscopy.
See ../PHYSICS-MODEL.md for sources, assumptions and limitations.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit


def gem_conductivity(sigma_low, sigma_high, fraction=0.20, threshold=0.30,
                     exponent=2.0):
    """Solve equal-exponent GEM exactly with cancellation-safe quadratic roots.

    Supports broadcast arrays. The inputs must be positive finite phase
    conductivities, 0 <= fraction <= 1, 0 < threshold < 1, and exponent > 0.
    Equal phase values and pure-phase endpoints are handled exactly. This is
    not an arithmetic mixture and not an empirical output-only transform.
    """
    sl, sh, f, fc, w = np.broadcast_arrays(*[
        np.asarray(v, dtype=float) for v in
        (sigma_low, sigma_high, fraction, threshold, exponent)])
    if any(np.any(~np.isfinite(v)) for v in (sl, sh, f, fc, w)):
        raise ValueError("GEM parameters must be finite")
    if np.any(sl <= 0) or np.any(sh <= 0):
        raise ValueError("Phase conductivities must be positive")
    if np.any((f < 0) | (f > 1)) or np.any((fc <= 0) | (fc >= 1)):
        raise ValueError("Invalid GEM fractions or threshold")
    if np.any(w <= 0):
        raise ValueError("GEM exponent must be positive")
    # Normalize both transformed phase values to avoid overflow. For the
    # documented parameter domain this also avoids underflow at high contrast.
    log_scale = np.maximum(np.log(sl), np.log(sh))
    l = np.exp((np.log(sl) - log_scale) / w)
    h = np.exp((np.log(sh) - log_scale) / w)
    a = (1.0 - fc) / fc
    b = (1.0 - f) * (a * l - h) + f * (a * h - l)
    root = np.hypot(b, 2.0 * np.sqrt(a * l * h))
    x = np.empty_like(b)
    positive = b >= 0
    np.divide(b + root, 2.0 * a, out=x, where=positive)
    np.divide(2.0 * l * h, root - b, out=x, where=~positive)
    with np.errstate(divide="ignore", over="raise", invalid="raise"):
        result = np.exp(log_scale + w * np.log(x))
    result = np.where(f == 0, sl, np.where(f == 1, sh, result))
    result = np.where(sl == sh, sl, result)
    # Round-off must not put an effective conductivity outside its phase bounds.
    result = np.clip(result, np.minimum(sl, sh), np.maximum(sl, sh))
    return float(result) if result.ndim == 0 else result


def gem_residual(sigma_effective, sigma_low, sigma_high, fraction=0.20,
                 threshold=0.30, exponent=2.0):
    """Dimensionless residual of the original implicit GEM equation."""
    se, sl, sh, f, fc, w = np.broadcast_arrays(*[
        np.asarray(v, dtype=float) for v in
        (sigma_effective, sigma_low, sigma_high, fraction, threshold, exponent)])
    a = (1.0 - fc) / fc
    # Ratios avoid absolute powered conductivities and their units.
    l = np.exp((np.log(sl) - np.log(se)) / w)
    h = np.exp((np.log(sh) - np.log(se)) / w)
    return (1.0 - f) * (l - 1.0) / (l + a) + f * (h - 1.0) / (h + a)


def _number(row: Mapping, *names, default=math.nan):
    for name in names:
        value = row.get(name)
        if value is None or isinstance(value, (str, dict, list, tuple)):
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            return value
    return default


def _dose(row, *names):
    """Saturating *reported dose coordinate*, not dry-film volume fraction."""
    c = _number(row, *names)
    return c / (5.0 + c) if math.isfinite(c) and c > 0 else 0.0


def _thermal_dose(row):
    """Dimensionless Arrhenius exposure, only if both T and duration exist."""
    temp_c = _number(row, "anneal_temperature_C")
    seconds = _number(row, "anneal_duration_s")
    if not (math.isfinite(temp_c) and math.isfinite(seconds)) or seconds <= 0:
        return 0.0
    kelvin = temp_c + 273.15
    if kelvin <= 0:
        return 0.0
    # 0.35 eV is an explicitly declared sensitivity prior, not an identified
    # PEDOT reaction activation energy. Saturation prevents extreme extrapolation.
    log_exposure = math.log(seconds / 60.0) - 0.35 / 8.617333262e-5 * (
        1.0 / kelvin - 1.0 / 403.15)
    exposure = math.exp(float(np.clip(log_exposure, -60.0, 30.0)))
    return exposure / (1.0 + exposure)


def _centered_temperature(row, *names):
    c = _number(row, *names)
    return float(np.clip((c - 100.0) / 100.0, -1.5, 3.0)) if math.isfinite(c) else 0.0


def _speed(row, *names):
    speed = _number(row, *names)
    # m/s -> relative to 1 mm/s; no rpm conversion is permitted.
    return float(np.clip(math.log1p(speed / 0.001) / math.log(11.0), 0.0, 3.0)) \
        if math.isfinite(speed) and speed >= 0 else 0.0


FEATURE_NAMES = (
    "intercept", "dmso_saturation", "eg_saturation", "acid_post_present",
    "polar_solvent_post_present", "pss_mass_ratio_coordinate", "thermal_exposure",
    "coating_temperature_context", "coating_speed_context", "post_temperature_context",
    "post_speed_context", "grade_ph1000", "grade_ph500", "grade_ai4083",
    "grade_baytron_p", "out_of_plane_context",
)


def process_design(features: Sequence[Mapping]):
    """Fixed, low-dimensional, outcome/ID-free recipe coordinates.

    Absent inputs have no explicit contribution; they are *not* claimed zero
    concentration/room temperature in the source data. Downstream residual
    models should retain missingness indicators separately.
    """
    out = np.zeros((len(features), len(FEATURE_NAMES)), dtype=float)
    for i, row in enumerate(features):
        ratio = _number(row, "reported_pss_pedot_mass_ratio")
        ratio_coordinate = math.log1p(min(max(ratio, 0.0), 100.0)) / math.log(31.0) \
            if math.isfinite(ratio) and ratio >= 0 else 0.0
        direction = str(row.get("measurement_direction", row.get("direction", ""))).lower()
        out_of_plane = bool(_number(row, "measurement_out_of_plane", default=0)) or \
            direction in ("out_of_plane", "out-of-plane", "perpendicular", "vertical")
        out[i] = [
            1.0, _dose(row, "dmso_dose_volpct", "dmso_volpct"),
            _dose(row, "eg_dose_volpct", "eg_volpct"),
            float(bool(_number(row, "acid_post_present", default=0))),
            float(bool(_number(row, "polar_solvent_post_present", default=0))),
            ratio_coordinate, _thermal_dose(row),
            _centered_temperature(row, "coating_temperature_C"),
            _speed(row, "coating_speed_m_s"),
            _centered_temperature(row, "post_temperature_C"),
            _speed(row, "post_speed_m_s"),
            float(bool(_number(row, "grade_ph1000", default=0))),
            float(bool(_number(row, "grade_ph500", default=0))),
            float(bool(_number(row, "grade_ai4083", default=0))),
            float(bool(_number(row, "grade_baytron_p", default=0))),
            float(out_of_plane),
        ]
    return out


def hopping_coordinate(features: Sequence[Mapping], reference_temperature_K=298.15):
    """VRH/NNH temperature coordinate only for independently declared regimes.

    Missing regime or measurement T gives zero correction and active=False.
    No regime is guessed from conductivity or from annealing temperature.
    """
    exponents = {"vrh3d": 0.25, "vrh2d": 1.0 / 3.0, "es_vrh": 0.5,
                 "nearest_neighbor": 1.0, "nnh": 1.0}
    coordinate = np.zeros(len(features), dtype=float)
    active = np.zeros(len(features), dtype=bool)
    for i, row in enumerate(features):
        regime = str(row.get("transport_regime", row.get("hopping_regime", ""))).lower()
        temperature = _number(row, "measurement_temperature_K")
        if regime not in exponents or not math.isfinite(temperature) or temperature <= 0:
            continue
        coordinate[i] = float(np.clip(
            (reference_temperature_K / temperature) ** exponents[regime] - 1.0,
            -1.0, 50.0))
        active[i] = True
    return coordinate, active


class GreyBoxTransport:
    """Fit one sparse process closure and solve GEM; no per-row free parameters.

    Fit targets must be log10 conductivity in S/cm. Only the passed training
    rows are read. Parameters and priors can be inspected with ``summary``.
    ``components`` exposes inferred quantities explicitly labeled effective.
    """
    def __init__(self, *, fraction=0.20, threshold=0.30, exponent=2.0,
                 sigma_high=20000.0, low_log10_min=-12.0, regularization=0.01,
                 max_nfev=350, enable_hopping=True, random_state=20260919):
        gem_conductivity(1.0, sigma_high, fraction, threshold, exponent)
        if fraction >= threshold:
            raise ValueError("Default one-channel closure requires fraction < threshold")
        if low_log10_min >= math.log10(sigma_high):
            raise ValueError("Invalid low-phase conductivity interval")
        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        self.fraction = float(fraction)
        self.threshold = float(threshold)
        self.exponent = float(exponent)
        self.sigma_high = float(sigma_high)
        self.low_log10_min = float(low_log10_min)
        self.regularization = float(regularization)
        self.max_nfev = int(max_nfev)
        self.enable_hopping = bool(enable_hopping)
        self.random_state = int(random_state)
        self.coef_ = None

    def _compute(self, design, hopping, coef):
        z = design @ coef[:len(FEATURE_NAMES)]
        high_log = math.log10(self.sigma_high)
        low_log_reference = self.low_log10_min + (
            high_log - self.low_log10_min) * expit(z)
        low_log = np.clip(low_log_reference - coef[-1] * hopping,
                          self.low_log10_min, high_log)
        low_sigma = np.power(10.0, low_log)
        sigma = gem_conductivity(low_sigma, self.sigma_high, self.fraction,
                                 self.threshold, self.exponent)
        return np.log10(sigma), low_sigma, low_log_reference, z

    def fit(self, features: Sequence[Mapping], y, sample_weight=None):
        features = list(features)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(features) or len(target) == 0:
            raise ValueError("features/y must have equal nonzero length")
        if not np.all(np.isfinite(target)):
            raise ValueError("Training target must contain only finite log10 S/cm")
        weights = np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != target.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid sample weights")
        weights = weights / weights.mean()
        design = process_design(features)
        hopping, active = hopping_coordinate(features)
        if not self.enable_hopping:
            hopping[:] = 0.0
            active[:] = False
        n = len(FEATURE_NAMES)
        lower = np.full(n + 1, -4.0)
        upper = np.full(n + 1, 4.0)
        lower[0], upper[0] = -7.0, 7.0
        # Local, saturating nonnegative solvent prior; no unbounded linear gains.
        lower[1:3] = 0.0
        upper[5] = 0.0  # More explicitly reported PSS does not help this baseline.
        lower[-1], upper[-1] = 0.0, 8.0
        prior = np.zeros(n + 1)
        prior[1:3] = 0.25
        prior[5] = -0.1
        prior[-1] = 0.5 if active.any() else 0.0
        # Initialization uses only this training set. It is not a fitted
        # per-sample inverse GEM value; just one common starting intercept.
        intercept_grid = np.linspace(-6.0, 6.0, 301)
        grid_design = np.zeros((len(intercept_grid), n))
        grid_design[:, 0] = intercept_grid
        init_probe = np.zeros(n + 1)
        init_probe[0] = 1.0
        grid_y = self._compute(grid_design, np.zeros(len(intercept_grid)), init_probe)[0]
        prior[0] = intercept_grid[np.argmin(np.abs(grid_y - np.median(target)))]
        penalty_mask = np.ones(n + 1)
        penalty_mask[0] = 0.0
        if not active.any():
            # The dormant coefficient is regularized exactly to zero.
            prior[-1] = 0.0

        def objective(coef):
            prediction = self._compute(design, hopping, coef)[0]
            residual = np.sqrt(weights) * (prediction - target)
            penalty = math.sqrt(len(target) * self.regularization) * penalty_mask * (coef - prior)
            return np.concatenate([residual, penalty])

        # Two deterministic starts are enough for this small, smooth closure.
        candidates = []
        for shift in (0.0, 0.6):
            initial = np.clip(prior + np.r_[shift, np.zeros(n)], lower + 1e-8, upper - 1e-8)
            result = least_squares(objective, initial, bounds=(lower, upper),
                                   max_nfev=self.max_nfev, loss="linear", ftol=1e-9,
                                   xtol=1e-9, gtol=1e-9)
            candidates.append(result)
        best = min(candidates, key=lambda item: np.dot(item.fun, item.fun))
        self.coef_ = best.x.copy()
        if not active.any():
            self.coef_[-1] = 0.0
        self.fit_info_ = {
            "n_training_examples": len(target), "optimizer_success": bool(best.success),
            "optimizer_status": int(best.status), "optimizer_message": str(best.message),
            "optimizer_evaluations": int(best.nfev),
            "training_rmse_log10": float(np.sqrt(np.mean((self._compute(design, hopping, self.coef_)[0] - target) ** 2))),
            "training_hopping_active": int(active.sum()),
            "shared_fitted_parameters": int(n + bool(active.any())),
        }
        return self

    def components(self, features: Sequence[Mapping]):
        if self.coef_ is None:
            raise RuntimeError("Fit GreyBoxTransport before predicting")
        features = list(features)
        design = process_design(features)
        hopping, active = hopping_coordinate(features)
        if not self.enable_hopping:
            hopping[:] = 0
            active[:] = False
        prediction, sigma_low, low_log_reference, z = self._compute(design, hopping, self.coef_)
        return {"prediction_log10_S_cm": prediction,
                "effective_low_phase_S_cm": sigma_low,
                "effective_low_phase_reference_log10_S_cm": low_log_reference,
                "shared_process_score": z, "hopping_active": active,
                "hopping_coordinate": hopping,
                "gem_equation_residual": gem_residual(10.0 ** prediction, sigma_low,
                    self.sigma_high, self.fraction, self.threshold, self.exponent)}

    def predict(self, features: Sequence[Mapping]):
        return self.components(features)["prediction_log10_S_cm"]

    def summary(self):
        return {
            "model": "GEM transport with sparse shared process closure",
            "version": "1.0.0", "target": "log10 conductivity [S/cm]",
            "assumption_status": "effective modeling priors; not independently identified microstructure",
            "fixed_priors": {"high_phase_S_cm": self.sigma_high,
                "effective_high_phase_fraction": self.fraction, "percolation_threshold": self.threshold,
                "critical_exponent": self.exponent, "low_phase_log10_min": self.low_log10_min,
                "anneal_activation_energy_eV": 0.35, "anneal_reference_temperature_K": 403.15,
                "anneal_reference_duration_s": 60.0, "dose_half_saturation_reported_volpct": 5.0,
                "measurement_reference_temperature_K": 298.15},
            "regularization": self.regularization, "max_nfev": self.max_nfev,
            "coefficients": None if self.coef_ is None else dict(zip(
                FEATURE_NAMES + ("hopping_strength_log10",), map(float, self.coef_))),
            "fit": getattr(self, "fit_info_", None),
            "forbidden_inputs": ["source_id", "experiment_id", "sample_label", "observed_conductivity",
                "observed_coverage", "observed_thickness", "fitted_target_transport_parameters"],
            "limitations": ["Fixed effective geometry and phase conductivity need sensitivity/independent checks.",
                "Missing recipe inputs give no term, not a claim of zero concentration.",
                "Mass-ratio coordinate is never used as GEM volume fraction.",
                "Reported dose percentages can have different denominator conventions.",
                "Heating exposure uses max temperature and summed durations if that is all inputs provide.",
                "VRH/NNH applies only for explicitly supplied regime and measurement temperature.",
                "A bounded residual can improve empirical fit without validating the mechanism."],
        }
