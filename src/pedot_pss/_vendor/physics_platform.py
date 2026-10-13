"""Polybot platform branch: process-GEM film state + film-formation (dewetting) sub-model.

Drop-in replacement for ``original_polybot_clean.OriginalPolybotClean`` (same ``fit(raw, y)`` /
``predict_values(raw)`` interface, n x 5 output ``[total, physical, weighted_physical, residual, 0]``,
total = weighted_physical + residual). Inputs are the 24-field experiment dicts; only the nine planned
controls (and deterministic physics computed from them) reach the model.

Physics structure (two film states, squared-error-optimal expected value):

    y_hat(x) = (1 - p(x)) * [ w * GEM(x) + r(x) ]  +  p(x) * mu_dewet

* GEM(x): the unchanged process-GEM closure (``_prepare_rows(scientific, .35, "polybot")`` +
  ``GreyBoxTransport(max_nfev=150, enable_hopping=False, random_state=20261009)``) fitted on TRAIN rows.
  By default it is fitted on the TRAIN rows that are in the percolating-film state (log10 sigma above
  ``film_threshold_log10``), because the GEM describes a continuous PEDOT-rich film; ``gem_fit="all"``
  reproduces fitting on every TRAIN row.
* r(x): bounded ML residual of the film state (SVR on the controls + film-formation coordinates),
  fitted on the same film-state TRAIN rows against y - w*GEM.
* p(x): probability that the film did not form a percolating layer (dewetted / non-percolating film).
  Film-formation coordinates from the controls (see ``film_coordinates``): the blade-coating regime number
  xi1 = log10[Ca^(2/3) v1 / E(T1)] (Landau-Levich entrainment vs evaporative meniscus deposition, with the
  Antoine water flux at the stage temperature), the wash contact time 20 mm / v2 and the wash stage
  temperature T2. Two Gaussian-process classifiers (ARD) are fitted on the DMSO-containing TRAIN inks: a
  film-formation expert on xi1 and a solvent-wash expert on (log wash contact time, T2); their log-odds are
  combined as a product of experts (naive-Bayes evidence sum minus the prior log-odds), then sharpened by
  1/dewet_temperature. The failure channel is gated by DMSO: without DMSO it is inactive (p = 0).
  Optional privileged supervision (``privileged={"coverage": ..., "coverage_sd": ...}`` aligned with
  the TRAIN rows; TRAIN-only, never a prediction input) distils a coverage teacher into the classifier.
* mu_dewet: the non-percolating film level = mean log10 sigma of the TRAIN dewetted films.

Column semantics: physical = GEM(x); weighted_physical = (1-p) w GEM(x) + p mu_dewet (two-state physical
mixture); residual = (1-p) r(x).
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .clean_inputs import canonicalize, feature_view
from .calibrated_hybrid import _prepare_rows
from .hybrid import _physics

KEYS = (
    "pb_dmso_volpct_reported", "pb_eg_volpct_reported",
    "pb_first_coating_speed_mm_s", "pb_first_coating_stage_temperature_C",
    "pb_post_speed_mm_s", "pb_post_stage_temperature_C",
    "pb_post_methanol_volpct", "pb_post_ethanol_volpct", "pb_post_water_volpct",
)
SUBSTRATE_LENGTH_MM = 20.0      # 2 x 2 cm Si/SiO2 pieces
PH1000_SOLIDS_VOLPCT = 1.1      # ~1.1 wt% solids in PH1000
STAGE_DRY_S = 60.0


def eligible_scientific(row):
    return bool(row.get("grade_ph1000") == 1 and row.get("deposition_blade_coating") == 1
                and row.get("geometry_film") == 1 and row.get("substrate_silicon") == 1
                and all(isinstance(row.get(k), (float, int, np.number)) and np.isfinite(row[k]) for k in KEYS))


# ------------------------------------------------------------------ physical property correlations
def _antoine_kpa(T, A, B, C):
    return 10 ** (A - B / (C + np.asarray(T, float))) * 0.133322


def psat_kpa(T, solvent):
    """Saturation vapour pressure (kPa), T in degC. Antoine (mmHg) for water/MeOH/EtOH; Clausius-Clapeyron
    anchored at the normal boiling point for DMSO (dH_vap 50 kJ/mol) and EG (61 kJ/mol)."""
    if solvent == "water":
        return _antoine_kpa(T, 8.07131, 1730.63, 233.426)
    if solvent == "meoh":
        return _antoine_kpa(T, 8.08097, 1582.271, 239.726)
    if solvent == "etoh":
        return _antoine_kpa(T, 8.20417, 1642.89, 230.300)
    Tb, dH = {"dmso": (462.15, 50e3), "eg": (470.45, 61e3)}[solvent]
    return 101.325 * np.exp(-dH / 8.314 * (1.0 / (np.asarray(T, float) + 273.15) - 1.0 / Tb))


def water_viscosity_mPas(T):
    """Vogel correlation for water viscosity (mPa s)."""
    return 2.414e-2 * 10 ** (247.8 / (np.asarray(T, float) + 273.15 - 140.0))


_WASH = {  # bp degC, surface tension mN/m, molar volume cm3/mol
    "meoh": (64.7, 22.7, 40.7), "etoh": (78.4, 22.1, 58.5), "h2o": (100.0, 72.0, 18.0)}

FILM_COORDINATES = ("xi1", "logCa1", "logE1", "tdry1", "phi_hb", "t_hb", "marang",
                    "xi2", "logCa2", "logE2", "sup2", "gam2", "log_t1", "log_t2")


def film_coordinates(matrix):
    """Film-formation coordinates from the nine controls (columns ordered as KEYS). Returns n x 14."""
    m = np.asarray(matrix, float)
    dm, eg, v1, T1, v2, T2 = (m[:, i] for i in range(6))
    fr = m[:, 6:9] / np.maximum(m[:, 6:9].sum(1, keepdims=True), 1e-9)
    hb = dm + eg
    phi_w = 1.0 - hb / 100.0
    gam_w = 75.6 - 0.14 * T1                                   # water surface tension (mN/m)
    gam_ink = phi_w * gam_w + dm / 100.0 * 43.0 + eg / 100.0 * 48.0
    mu_ink = water_viscosity_mPas(T1) * (1.0 + 0.03 * hb)       # mPa s
    Ca1 = mu_ink * v1 * 1e-3 / gam_ink                         # (mPa s)(mm/s)/(mN/m) -> dimensionless x1e-3
    E1 = psat_kpa(T1, "water") / (T1 + 273.15)                 # diffusion-limited water flux ~ p_sat/T
    xi1 = (2.0 / 3.0) * np.log10(Ca1) + np.log10(v1) - np.log10(E1)   # log10 h_LL/h_evap (+const)
    h_wet = 50.0 * np.clip(Ca1 / 1e-4, 1e-3, None) ** (2.0 / 3.0)
    tdry1 = np.log10(h_wet / (E1 * 1e3)) - np.log10(SUBSTRATE_LENGTH_MM / v1)
    phi_hb = hb / (hb + PH1000_SOLIDS_VOLPCT)                  # residual high-boiler fraction after water loss
    t_hb = np.log10(1e-3 + (dm / psat_kpa(T1, "dmso") + eg / psat_kpa(T1, "eg")) * (h_wet / 50.0) / STAGE_DRY_S)
    gam_hb = np.where(hb > 0, (dm * 43.0 + eg * 48.0) / np.maximum(hb, 1e-9), gam_w)
    marang = (gam_w - gam_hb) * hb / 100.0                    # Marangoni driving force on enrichment
    bp = fr @ np.array([_WASH[s][0] for s in ("meoh", "etoh", "h2o")])
    gam2 = fr @ np.array([_WASH[s][1] for s in ("meoh", "etoh", "h2o")])
    mol = fr / np.array([_WASH[s][2] for s in ("meoh", "etoh", "h2o")])
    x = mol / np.maximum(mol.sum(1, keepdims=True), 1e-12)
    p2 = x[:, 0] * psat_kpa(T2, "meoh") + x[:, 1] * psat_kpa(T2, "etoh") + x[:, 2] * psat_kpa(T2, "water")
    mu2 = 0.55 * (1.0 + 0.5 * fr[:, 1]) * water_viscosity_mPas(T2) / water_viscosity_mPas(25.0)
    Ca2 = mu2 * v2 * 1e-3 / gam2
    E2 = p2 / (T2 + 273.15)
    xi2 = (2.0 / 3.0) * np.log10(Ca2) + np.log10(v2) - np.log10(E2)
    return np.column_stack([xi1, np.log10(Ca1), np.log10(E1), tdry1, phi_hb, t_hb, marang, xi2, np.log10(Ca2),
                            np.log10(E2), T2 - bp, gam2, np.log10(SUBSTRATE_LENGTH_MM / v1),
                            np.log10(SUBSTRATE_LENGTH_MM / v2)])


def _named(matrix, names):
    coords = film_coordinates(matrix)
    raw = {"dmso": 0, "eg": 1, "v1": 2, "T1": 3, "v2": 4, "T2": 5, "meoh": 6, "etoh": 7, "h2o": 8}
    cols = []
    for n in names:
        cols.append(matrix[:, raw[n]] if n in raw else coords[:, FILM_COORDINATES.index(n)])
    return np.column_stack(cols).astype(float)


class PhysicsPlatformPolybot:
    """Two-state (percolating GEM film / dewetted film) Polybot platform branch. See module docstring."""
    version = "polybot-physics-platform-gem-dewet-1012"

    def __init__(self, gem_weight=1.0, gem_fit="film", film_threshold_log10=0.0,
                 dewet_structure="poe", dewet_experts=(("xi1",), ("log_t2", "T2")), rule_coord="v1",
                 dewet_coords=("xi1", "log_t2", "T2"), dewet_temperature=0.6, dewet_restarts=3,
                 residual_coords=("dmso", "eg", "v1", "T1", "v2", "T2", "meoh", "etoh", "h2o") + FILM_COORDINATES,
                 svr_C=3.0, svr_gamma=0.05, svr_epsilon=0.05, use_privileged=True, random_state=20261009):
        self.gem_weight, self.gem_fit, self.film_threshold_log10 = gem_weight, gem_fit, film_threshold_log10
        self.dewet_structure, self.rule_coord = dewet_structure, rule_coord
        self.dewet_experts = tuple(tuple(e) for e in dewet_experts)
        self.dewet_coords, self.dewet_temperature, self.dewet_restarts = tuple(dewet_coords), dewet_temperature, dewet_restarts
        self.residual_coords = tuple(residual_coords)
        self.svr_C, self.svr_gamma, self.svr_epsilon = svr_C, svr_gamma, svr_epsilon
        self.use_privileged, self.random_state = use_privileged, random_state

    @staticmethod
    def _inputs(raw):
        clean = [canonicalize(row) for row in raw]
        scientific = [feature_view(row) for row in clean]
        if not all(eligible_scientific(row) for row in scientific):
            raise ValueError("Polybot physics route requires PH1000/blade/silicon/film and complete nine controls")
        matrix = np.asarray([[row[k] for k in KEYS] for row in scientific], dtype=float).reshape(len(raw), 9)
        prepared, _ = _prepare_rows(scientific, .35, "polybot")   # unchanged process-GEM preparation
        return matrix, prepared

    # ---------------------------------------------------------------- dewetting sub-model
    def _expert_sets(self):
        """List of (kind, coordinate names). kind 'gp' = GP classifier expert; 'rule' = fold-fitted threshold
        criterion 'DMSO and coordinate <= c -> dewetted' (slow blade coating / thin evaporative-regime film)."""
        st = self.dewet_structure
        if st == "poe":
            return [("gp", e) for e in self.dewet_experts]
        if st == "joint":
            return [("gp", self.dewet_coords)]
        if st in ("rule", "rule_hard"):
            return [("rule", (self.rule_coord,))]
        if st == "rule_poe":   # threshold criterion (first stage) x GP wash expert
            return [("rule", (self.rule_coord,))] + [("gp", e) for e in self.dewet_experts[1:]]
        raise ValueError(f"unknown dewet_structure {st!r}")

    @staticmethod
    def _fit_rule(x, t):
        """Threshold c over the TRAIN DMSO-ink values minimising TRAIN errors of 'x <= c -> failed'
        (ties -> smallest c); returns Laplace-smoothed TRAIN failure fractions below/above c."""
        best = None
        for c in np.unique(x):
            err = int(((x <= c).astype(int) != t).sum())
            if best is None or err < best[0]:
                best = (err, c)
        c = float(best[1]); below = x <= c
        pb = (t[below].sum() + 0.5) / (below.sum() + 1.0)
        pa = (t[~below].sum() + 0.5) / ((~below).sum() + 1.0)
        return ("rule", c, float(pb), float(pa), int(best[0]))

    def _fit_expert(self, X, t, privileged_soft):
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if privileged_soft is not None:
                # LUPI-style distillation: student GP regression on logit of (hard + teacher soft)/2 labels.
                target = logit(np.clip(.5 * t + .5 * privileged_soft, .02, .98))
                k = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(.05)
                model = GaussianProcessRegressor(k, random_state=0, n_restarts_optimizer=self.dewet_restarts).fit(Xs, target)
                return ("gp_distilled", scaler, model)
            k = ConstantKernel(1.0) * RBF(np.ones(X.shape[1]), (1e-2, 1e3))
            model = GaussianProcessClassifier(k, random_state=0, n_restarts_optimizer=self.dewet_restarts).fit(Xs, t)
            return ("gpc", scaler, model)

    def _expert_proba(self, expert, X):
        if expert[0] == "rule":
            _, c, pb, pa, _ = expert
            below = X[:, 0] <= c
            if self.dewet_structure == "rule_hard":
                return np.where(below, 1 - 1e-4, 1e-4)
            return np.where(below, pb, pa)
        kind, scaler, model = expert
        Xs = scaler.transform(X)
        q = expit(model.predict(Xs)) if kind == "gp_distilled" else model.predict_proba(Xs)[:, 1]
        return np.clip(q, 1e-4, 1 - 1e-4)

    def _fit_dewet(self, matrix, failed, privileged):
        """Dewetting channel, DMSO-gated. 'poe': product of experts (naive-Bayes log-odds sum) of a
        first-stage film-formation expert (blade-coating regime) and a solvent-wash expert; 'joint': one ARD GP."""
        dmso = matrix[:, 0] > 0
        self.dewet_mode_, self.dewet_base_rate_, self.experts_ = "inactive", 0.0, []
        if failed.sum() == 0:
            return
        n_dm, n_fail = int(dmso.sum()), int((failed & dmso).sum())
        self.dewet_base_rate_ = (n_fail + 0.5) / (n_dm + 1.0) if n_dm else 0.0
        if n_fail < 2 or n_dm - n_fail < 2:
            self.dewet_mode_ = "base_rate"
            return
        t = failed[dmso].astype(int)
        soft = None
        if self.use_privileged and privileged is not None:
            cov = np.asarray(privileged["coverage"], float)[dmso] / 100.0
            csd = np.asarray(privileged.get("coverage_sd", np.zeros(len(failed))), float)[dmso] / 100.0
            teacher = LogisticRegression(C=1.0, max_iter=5000).fit(np.c_[cov, csd], t)   # TRAIN-only teacher
            soft = np.clip(teacher.predict_proba(np.c_[cov, csd])[:, 1], .02, .98)
        for kind, names in self._expert_sets():
            X = _named(matrix[dmso], names)
            fitted = self._fit_rule(X[:, 0], t) if kind == "rule" else self._fit_expert(X, t, soft)
            self.experts_.append((names, fitted))
        self.prior_logit_ = float(logit(np.clip(t.mean(), 1e-3, 1 - 1e-3)))
        self.dewet_mode_ = self.dewet_structure + ("_distilled" if soft is not None else "")

    def _dewet_probability(self, matrix):
        p = np.zeros(len(matrix))
        dmso = matrix[:, 0] > 0
        if self.dewet_mode_ == "inactive" or not dmso.any():
            return p
        if self.dewet_mode_ == "base_rate":
            p[dmso] = self.dewet_base_rate_
            return p
        logits = [logit(self._expert_proba(expert, _named(matrix[dmso], names))) for names, expert in self.experts_]
        z = np.sum(logits, axis=0) - (len(logits) - 1) * self.prior_logit_
        if self.dewet_structure == "rule_hard":
            p[dmso] = (z > 0).astype(float)                       # hard call: failure-level or film-state
            return p
        p[dmso] = expit(z / float(self.dewet_temperature))     # sharpening of the Laplace-GP output
        return p

    # ---------------------------------------------------------------- fit / predict
    def fit(self, raw, y, privileged=None):
        matrix, prepared = self._inputs(raw)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(raw),) or not len(raw) or not np.isfinite(y).all():
            raise ValueError("Finite matching TRAIN targets required")
        if privileged is not None:
            for key, value in privileged.items():
                if np.asarray(value).shape != (len(raw),):
                    raise ValueError(f"privileged[{key!r}] must align with the TRAIN rows")
        film = y > self.film_threshold_log10
        failed = ~film
        gem_rows = np.flatnonzero(film) if (self.gem_fit == "film" and film.sum() >= 8) else np.arange(len(y))
        self.physics_ = _physics.GreyBoxTransport(max_nfev=150, enable_hopping=False, random_state=20261009)
        self.physics_.fit([prepared[i] for i in gem_rows], y[gem_rows])
        physical = self.physics_.predict(prepared)
        w = float(self.gem_weight)
        # film-state residual on the percolating TRAIN films
        fr = np.flatnonzero(film) if film.sum() >= 3 else np.arange(len(y))
        Z = _named(matrix, self.residual_coords)
        self.scaler_ = StandardScaler().fit(Z[fr])
        resid = y[fr] - w * physical[fr]
        self.res_offset_ = float(resid.mean())
        self.svr_ = SVR(C=self.svr_C, gamma=self.svr_gamma, epsilon=self.svr_epsilon).fit(
            self.scaler_.transform(Z[fr]), resid - self.res_offset_)
        # dewetted (non-percolating) film level and the dewetting sub-model
        self.mu_dewet_ = float(y[failed].mean()) if failed.any() else -1.3
        self._fit_dewet(matrix, failed, privileged)
        self.n_training_examples_ = len(raw)
        self.n_film_training_ = int(film.sum())
        return self

    def components(self, raw):
        matrix, prepared = self._inputs(raw)
        physical = self.physics_.predict(prepared)
        w = float(self.gem_weight)
        r = self.res_offset_ + self.svr_.predict(self.scaler_.transform(_named(matrix, self.residual_coords)))
        p = self._dewet_probability(matrix)
        return {"gem_log10": physical, "film_state_log10": w * physical + r, "film_residual": r,
                "p_dewet": p, "mu_dewet": self.mu_dewet_}

    def predict_values(self, raw):
        if not len(raw):
            return np.empty((0, 5))
        c = self.components(raw)
        w = float(self.gem_weight); p = c["p_dewet"]
        weighted_physical = (1.0 - p) * w * c["gem_log10"] + p * c["mu_dewet"]
        residual = (1.0 - p) * c["film_residual"]
        total = weighted_physical + residual
        return np.column_stack((total, c["gem_log10"], weighted_physical, residual, np.zeros(len(raw))))

    def predict(self, raw):
        return self.predict_values(raw)[:, 0]
