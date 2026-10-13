"""Input-only adapters to original acid, salt and outer-PEO mathematical cores.

No original CV runner, source manifest, saved OOF, targets or filesystem data
is read. Fits receive explicit TRAIN targets; forwards do not accept targets.
Unknown input/protocol/basis stays inactive and retains the supplied parent.
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np
from . import domain_transport_v2 as transport
from .clean_inputs import canonicalize, feature_view, original_view
from . import acid_interface_closure as acid
from . import acid_process_gate as acid_gate
from .acid_process_gate_fit import AcidProcessGateCalibrator
from . import peo_composite_transport as peo
from . import peo_cross_host as cross
from .salt_protocol_inputs import prepare_salt_protocol_inputs
from .salt_lowphase_boundary import salt_contact_extent, lowphase_removal_volume_balance
from .phase_fraction_forward import solve_phase_state


VERSION = "original-special-physics-input-adapters-v1-1012"
# Published, reviewed starting-material property table, not target-derived.
CATION_SOFTNESS = {"AgNO3": .18, "CuCl2": .38, "InCl3": .48,
                  "LiCl": -1.02, "NaCl": -.75, "MgCl2": -.41, "NiCl2": -.11}
# Exactly the original salt increment's explicit six parameter hypotheses.
SALT_PARAMETERS = {"K_ref_M": .074, "softness_slope": 1.,
    "k_ref_per_s": math.log(2.) / 300., "Ea_eV": 0., "T_ref_K": 413.15,
    "max_removed_fraction": .5}
_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ACID_NAMES = {key.replace("_", " "): key for key in acid.SUPPORTED_MAX_M}


def _text(value):
    return (unicodedata.normalize("NFKC", value).replace("µ", "u").replace("μ", "u")
            if isinstance(value, str) else "")


def _extra(row, name, units):
    values = [float(e["value"]) for e in row["extra_scalar_features"]
              if e["name"] == name and e.get("unit") in units]
    return values[0] if values and len(set(values)) == 1 else None


def _near(a, b):
    return a is not None and math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9)


def acid_inputs(rows):
    """Typed applied-solution molarity and explicit 140C hotdrop/dry/rinse gate.

    Unit case matters (mM vs M). No concentration is inferred from grade,
    neatness, array maximum, arbitrary evidence, sample label or film state.
    The local original endpoint reference and support limits remain unchanged.
    """
    clean = [canonicalize(row) for row in rows]
    typed, admitted, diagnostics = [], [], []
    for row in clean:
        post = _text(row["post_treatment"])
        identities = {identity for name, identity in _ACID_NAMES.items()
                      if re.search(r"\b" + re.escape(name) + r"\b", post, re.I)}
        choices = []
        # Explicit chemistry within the applied post-treatment input only.
        for match in re.finditer(rf"(?P<v>{_NUM})\s*(?P<u>mmol/L|mol/L|mM|M)\s+(?P<a>[A-Za-z -]+? acid)\b", post):
            name = match.group("a").strip().lower()
            if name in _ACID_NAMES:
                value = float(match.group("v")) * (.001 if match.group("u") in {"mM", "mmol/L"} else 1.)
                choices.append((_ACID_NAMES[name], value))
        choices = list(set(choices))
        feature = {}
        if len(choices) == 1 and identities == {choices[0][0]}:
            identity, value = choices[0]
            feature = {acid.IDENTITY_KEY: identity, acid.CONCENTRATION_KEY: value,
                       acid.HAS_KEY: 1., acid.POSITIVE_KEY: float(value > 0)}
        # A semantic event chain, not a complete article-specific string match.
        # It must certify applied acid -> 140C evaporation -> DI-water rinse ->
        # separate 140C redry. Cooling does not become a constant thermal hold.
        drop = re.search(r"\bdrop(?:ped)?\b", post, re.I)
        dries = list(re.finditer(r"\b(?:dry|dried|drying|redry|redried)\b", post, re.I))
        rinse = re.search(r"\brins(?:e|ed|ing)\b[^;]{0,40}\b(?:DI|deionized)\s+water\b", post, re.I)
        temperatures = list(re.finditer(r"140\s*(?:°\s*)?C\b", post, re.I))
        durations = list(re.finditer(r"(?:about\s+)?10\s*min\b", post, re.I))
        chain = bool(drop and len(dries) >= 2 and rinse and len(temperatures) >= 2
                     and len(durations) >= 2 and drop.start() < dries[0].start() < rinse.start() < dries[-1].start())
        if row["process_stages"]:
            stages = row["process_stages"]
            candidates = [i for i, s in enumerate(stages) if s["kind"] == "post_treat"
                and _near(s["temperature_C"], 140) and re.search(r"\bdrop", _text(s["method"]), re.I)]
            chain = False
            for i in candidates:
                subsequent = stages[i+1:]
                if len(subsequent) >= 3:
                    d1 = next((j for j, s in enumerate(subsequent) if s["kind"] == "dry"
                               and _near(s["temperature_C"], 140) and _near(s["duration_s"], 600)), None)
                    rinse_i = next((j for j, s in enumerate(subsequent) if s["kind"] == "rinse"
                                    and re.search(r"\b(?:DI|deionized)\s+water\b", _text(s["method"]), re.I)), None)
                    d2 = next((j for j, s in enumerate(subsequent) if s["kind"] == "dry"
                               and _near(s["temperature_C"], 140) and _near(s["duration_s"], 600)
                               and rinse_i is not None and j > rinse_i), None)
                    chain = d1 is not None and rinse_i is not None and d2 is not None and d1 < rinse_i < d2
                    if chain: break
        supported = acid.acid_coordinates([feature])[0]["has_concentration"]
        typed.append(feature)
        admitted.append(bool(chain and supported))
        diagnostics.append({"has_supported_typed_concentration": bool(supported),
                            "verified_local_hotdrop_protocol": bool(chain)})
    return {"clean": clean, "acid_rows": typed, "protocol_admitted": np.asarray(admitted, bool),
            "diagnostics": diagnostics, "protocol_scope": acid_gate.PROTOCOL_SCOPE}


class OriginalAcidProcess:
    """The original 3-parameter fitter and actual GEM/circuit re-solve."""
    def __init__(self, regularization=.02, max_nfev=120):
        self.regularization, self.max_nfev = regularization, max_nfev

    def fit(self, rows, y_train, *, base, physical_rows=None):
        inputs = acid_inputs(rows)
        physical = list(physical_rows) if physical_rows is not None else [feature_view(r) for r in inputs["clean"]]
        self.calibrator_ = AcidProcessGateCalibrator(regularization=self.regularization, max_nfev=self.max_nfev)
        self.calibrator_.fit(physical, inputs["acid_rows"], y_train, base=base,
                            protocol_admitted=inputs["protocol_admitted"])
        return self

    def components(self, rows, *, base, physical_rows=None):
        inputs = acid_inputs(rows)
        physical = list(physical_rows) if physical_rows is not None else [feature_view(r) for r in inputs["clean"]]
        return self.calibrator_.components(physical, inputs["acid_rows"], base=base,
                                          protocol_admitted=inputs["protocol_admitted"])


def salt_inputs(rows, *, cation_softness=CATION_SOFTNESS):
    """Original stage ledger: known salt contact/rinse/redry, all units bound."""
    return prepare_salt_protocol_inputs([original_view(canonicalize(r)) for r in rows],
                                        cation_softness=cation_softness)


def salt_forward(rows, log_low_query, reference_parts, gem_model, *, parameters=SALT_PARAMETERS,
                 cation_softness=CATION_SOFTNESS):
    """Reuse original extent/balance and actual GEM+circuit at predicted low state.

    The caller supplies its QUERY-predicted low phase and frozen effective
    rho/bypass; no target inverse is computed here. Original salt parameters
    are fixed hypotheses, not newly fitted or claimed identified kinetics.
    """
    coordinates = salt_inputs(rows, cation_softness=cation_softness)["coordinates"]
    removed = np.asarray(salt_contact_extent(**coordinates, **parameters), float)
    n = len(rows)
    original_f = np.full(n, gem_model.fraction)
    thresholds = np.full(n, gem_model.threshold)
    fractions = np.asarray(lowphase_removal_volume_balance(original_f, removed), float)
    result = solve_phase_state(log_low_query, reference_parts, gem_model,
                               fractions=fractions, thresholds=thresholds)
    unchanged = ~coordinates["active"] | (removed == 0)
    for key, values in reference_parts.items():
        if key in result and np.asarray(values).shape == (n,):
            result[key][unchanged] = np.asarray(values)[unchanged]
    result["salt_removed_low_fraction"] = removed
    result["salt_active"] = coordinates["active"].copy()
    return result


def _molecular_mass(text, polymer):
    match = re.search(rf"\b{polymer}(?:\d+)?\b[^;)]{{0,25}}?Mw\s*[=:]?\s*({_NUM})\s*(kg|g)\s*(?:/\s*mol|mol\s*-?1)", text, re.I)
    return float(match.group(1)) * (1000 if match.group(2).lower() == "kg" else 1) if match else None


def peo_inputs(rows):
    """Scientific host/fraction/protocol predicates; no RAW_PROFILE identity gate.

    The old PH1000-EG expert is limited to fresh unwashed drop-cast blends
    with the explicitly known original preparation quantities. Family1 requires
    explicit high-Mw host chemistry and the declared dry-blend PEO mass basis.
    Missing storage/basis/material facts do not mean zero or pure host.
    """
    records = []
    for raw in rows:
        row = canonicalize(raw); f = feature_view(row)
        grade = _text(row["polymer_grade"]); sequence = _text(row["process_sequence"])
        fresh = _near(_extra(row, "air_storage_time", {"s", "min", "h"}), 0)
        fresh = fresh or bool(re.search(r"\bas.fabricated\b|\bfreshly\s+prepared\b", sequence, re.I))
        unwashed = row["post_treatment"] is None and not re.search(r"\b(?:rinse|wash|acid\s+treat|extrud|thermocompress)\w*\b", sequence, re.I)
        record = {"active": False, "family": None, "phi_nominal": None, "peo_mass_fraction": None,
                  "reason": "unknown_or_unsupported_material_basis_or_preparation"}
        w = _extra(row, "blend_pedot_pss_dry_mass_percent", {"%"})
        if w is not None: w /= 100
        if (fresh and unwashed and f.get("grade_ph1000") == 1 and f.get("deposition_drop_casting") == 1
                and f.get("substrate_pi") == 1 and w is not None and 0 < w <= 1):
            matrix = w == 1 or (_near(_molecular_mass(grade, "PEO"), 200000)
                and re.search(r"\bdemineralized\s+water\b", sequence, re.I))
            dose = any(re.search(r"ethylene\s+glycol|\bEG\b", name, re.I)
                and amount is not None and re.search(r"5\.25\s*mL\b", amount)
                and re.search(r"100\s*mL\b", amount)
                for name, amount in zip(row["additive_names"], row["additive_amounts_raw"]))
            profile = (matrix and dose and _near(row["stock_solids_wt_percent"], 1.2)
                and _near(row["dry_temperature_C"], 70) and _near(row["dry_duration_min"], 45)
                and _near(row["anneal_temperature_C"], 120) and _near(row["anneal_duration_min"], 60)
                and re.search(r"800\s*uL\b", sequence, re.I)
                and (w == 1 or re.search(r"sonicat\w*\s+45\s*min\b", sequence, re.I)))
            if profile:
                record.update(active=True, family=0, peo_mass_fraction=1-w, reason="known_PH1000_EG_PEO_dry_blend_protocol")
        if not record["active"] and fresh and unwashed and f.get("deposition_drop_casting") == 1:
            mass = None
            for name, amount in zip(row["additive_names"], row["additive_amounts_raw"]):
                if re.search(r"\bPEO(?:\d+)?\b", name, re.I) and amount:
                    match = re.fullmatch(rf"\s*({_NUM})\s*wt%\s+of\s+PEO/\(PEO\+PEDOT:PSS\)\s+blend\s*", amount, re.I)
                    if match: mass = float(match.group(1))/100
            chemistry = (_near(_molecular_mass(grade, "PSS"), 1e6)
                and _near(_molecular_mass(grade, "PEO"), 8e6)
                and _near(f.get("reported_pss_pedot_mass_ratio"), 2.5)
                and any(re.search(r"\bDMSO\b", name, re.I) and amount
                    and re.fullmatch(r"2\s*wt%\s+in\s+aqueous\s+solution", amount, re.I)
                    for name, amount in zip(row["additive_names"], row["additive_amounts_raw"])))
            if (chemistry and mass is not None and 0 <= mass < 1
                    and _near(row["dry_temperature_C"], 70) and _near(row["measurement_temperature_C"], 25)):
                w = 1-mass
                record.update(active=True, family=1, peo_mass_fraction=mass,
                              reason="known_high_Mw_PEDOT_PSS_DMSO_PEO_dry_blend_protocol")
        if record["active"]:
            w = 1-record["peo_mass_fraction"]
            record["phi_nominal"] = (w/peo.DENSITIES[0])/(w/peo.DENSITIES[0]+(1-w)/peo.DENSITIES[1])
        records.append(record)
    return {"records": records, "active": np.asarray([r["active"] for r in records], bool),
            "phi": np.asarray([r["phi_nominal"] if r["active"] else np.nan for r in records]),
            "family": np.asarray([r["family"] if r["active"] else -1 for r in records], int),
            "peo_mass": np.asarray([r["peo_mass_fraction"] if r["active"] else np.nan for r in records])}


class OriginalPEOExpert:
    """Reuse original percolation/GEM and selected SHARED_MORPH fit/forward."""
    def __init__(self, old_law="percolation", cross_arm="SHARED_MORPH"):
        self.old_law, self.cross_arm = old_law, cross_arm

    def fit(self, rows, y_train):
        inputs = peo_inputs(rows); y = np.asarray(y_train, float)
        if y.shape != inputs["active"].shape or not np.isfinite(y).all():
            raise ValueError("Aligned finite TRAIN log10 targets required")
        old = inputs["active"] & (inputs["family"] == 0)
        self.old_fit_ = (peo.fit(inputs["phi"][old], y[old], law=self.old_law)
            if old.sum() >= 4 and len(np.unique(inputs["phi"][old])) >= 3 else None)
        active = inputs["active"]
        support = [len(np.unique(inputs["phi"][active & (inputs["family"] == h)])) for h in (0, 1)]
        self.cross_fit_ = (cross.fit(inputs["phi"][active], inputs["family"][active],
            inputs["peo_mass"][active], y[active], self.cross_arm)
            if active.sum() >= 8 and min(support) >= 2 else None)
        self.n_admitted_train_ = int(active.sum())
        return self

    def predict(self, rows, parent_log10):
        inputs = peo_inputs(rows); result = np.asarray(parent_log10, float).copy()
        if result.shape != inputs["active"].shape or not np.isfinite(result).all():
            raise ValueError("Aligned finite parent QUERY predictions required")
        if getattr(self, "cross_fit_", None) is not None:
            use = inputs["active"]
            if use.any():
                result[use] = cross.forward(inputs["phi"][use], inputs["family"][use],
                    inputs["peo_mass"][use], self.cross_arm, self.cross_fit_["parameters"])["prediction_log10_S_cm"]
        elif getattr(self, "old_fit_", None) is not None:
            use = inputs["active"] & (inputs["family"] == 0)
            if use.any():
                result[use] = peo.forward(inputs["phi"][use], law=self.old_law,
                    **self.old_fit_["parameters"])["prediction_log10_S_cm"]
        return result


def retention_status():
    return {"version": VERSION,
        "acid": "original 3-parameter fit and GEM/circuit forward exposed; requires supported molarity and certified local hotdrop/dry/rinse conditions",
        "salt": "original fixed extent and volume balance plus GEM/circuit solve exposed; no original fitter exists for its six fixed assumptions",
        "peo": "original percolation/GEM and SHARED_MORPH fit/forward exposed; input-sufficient scientific host/fraction/protocol admission only",
        "parameter_reuse": "Original formulas/property priors retained. Old source/observed-trained calibration parameters are not automatically reused; callers must fit training folds or pass a scientifically compatible fitted base.",
        "not_retained": "Source/profile-ID routing, literal whole-article RAW_PROFILE matching, observed morphology/assay statistics, old CV/export/manifest orchestration and any unimplemented learned correction head",
        "complete_original_M_retained": False}


def _values(value, n, name):
    a = np.asarray(value, float)
    if a.shape != (n, 5) or not np.isfinite(a).all():
        raise ValueError(name + " must be finite aligned Nx5 values")
    if np.max(np.abs(a[:, 0]-a[:, 2]-a[:, 3]), initial=0.) > 1e-8:
        raise ValueError(name + " total must equal weighted physics plus residual")
    return a.copy()


def _physical_rows(rows):
    clean = [canonicalize(row) for row in rows]
    scientific = [feature_view(row) for row in clean]
    # Reuse the already checked explicit-stage thermal alias handling.
    from .planned_thermal_inputs import physics_rows_for_stages
    return clean, physics_rows_for_stages(clean, scientific)


def _certificates(parts):
    result = {}
    for key in ("gem_equation_residual", "closure_residual"):
        if key in parts:
            result[key + "_max_abs"] = float(np.max(np.abs(parts[key]), initial=0.))
    for key in ("gem_phase_bound_violation", "domain_bound_violation"):
        if key in parts:
            result[key + "_count"] = int(np.count_nonzero(parts[key]))
    if "sigma_domain_S_cm" in parts:
        result["nonpositive_conductivity_count"] = int(np.count_nonzero(parts["sigma_domain_S_cm"] <= 0))
    return result


class OriginalSpecialPhysicsChain:
    """Fold-fit original core bridges, preserving M residual on local deltas.

    Acid/salt increments are computed with the SAME fitted prior/effective state
    and added to M's 0/1/2 columns, with the circuit increment added to column4.
    This is the original ensemble-increment composition; the final combined
    M+increment is not claimed to be one unified GEM solution. PEO is a complete
    replacement expert [Q,Q,Q,0,0], rather than inherited PEDOT residual.
    """
    def __init__(self, max_nfev=150, acid_max_nfev=120):
        self.max_nfev, self.acid_max_nfev = max_nfev, acid_max_nfev

    def fit(self, rawTRAIN, MTRAINvalues, yTRAIN, fold=0):
        clean, physical = _physical_rows(list(rawTRAIN))
        values = _values(MTRAINvalues, len(clean), "MTRAINvalues")
        target = np.asarray(yTRAIN, float)
        if target.shape != (len(clean),) or not len(clean) or not np.isfinite(target).all():
            raise ValueError("Nonempty finite aligned TRAIN targets required")
        self.fold_ = int(fold)
        # Ordinary supervised TRAIN physics target: no query outcome/state
        # inverse or previously saved retrospective source model is used.
        physical_target = target-values[:, 3]
        self.base_ = transport.DomainTransport(mode="anisotropic_bypass", coupling="joint",
            max_nfev=self.max_nfev, activation_energy_eV=.35,
            regularization=.03, gem_regularization=.01, random_state=20261012+self.fold_)
        self.base_.fit(physical, physical_target)
        self.acid_ = OriginalAcidProcess(max_nfev=self.acid_max_nfev)
        self.acid_.fit(clean, physical_target, base=self.base_, physical_rows=physical)
        self.peo_ = OriginalPEOExpert()
        self.peo_.fit(clean, target)
        self.prior_fit_n_ = len(clean)
        acid_input, salt_input, peo_input = acid_inputs(clean), salt_inputs(clean), peo_inputs(clean)
        self.state_ = {"version": VERSION, "fold": self.fold_, "n_train": len(clean),
            "prior_fit_n": self.prior_fit_n_, "prior_target": "TRAIN conductivity log10 minus MTRAIN residual column3",
            "prior": self.base_.summary(), "acid": self.acid_.calibrator_.summary(),
            "salt": {"fixed_parameters": dict(SALT_PARAMETERS), "fixed_cation_softness": dict(CATION_SOFTNESS),
                     "fit_interface": None, "training_admission": salt_input["audit"]},
            "peo": {"old_fit": self.peo_.old_fit_, "cross_fit": self.peo_.cross_fit_,
                    "n_admitted_train": self.peo_.n_admitted_train_},
            "admitted_train": {"acid_protocol": int(acid_input["protocol_admitted"].sum()),
                "salt": int(salt_input["coordinates"]["active"].sum()), "peo": int(peo_input["active"].sum())},
            "no_query_targets_used": True, "no_source_or_observed_features": True,
            "composition": "acid/salt add physical delta to M columns0/1/2 and circuit delta to4; preserve3; PEO expert replaces [Q,Q,Q,0,0]",
            "complete_original_M_retained": False}
        return self

    def predict_values(self, rawQUERY, MQUERYvalues):
        if not hasattr(self, "base_"):
            raise RuntimeError("Fit on the training fold before QUERY prediction")
        clean, physical = _physical_rows(list(rawQUERY))
        source = _values(MQUERYvalues, len(clean), "MQUERYvalues")
        result = source.copy()
        if not clean:
            return result, {"n_query": 0, "outside_changed": 0, "prior_fit_n": self.prior_fit_n_}
        baseline = self.base_.components(physical)
        acid_parts = self.acid_.components(clean, base=self.base_, physical_rows=physical)
        # Obtain the actual QUERY-predicted low state at the modified acid
        # process coordinate. Recompute the original core, never invert QUERY y.
        prepared, design, _, _ = transport._inputs(physical, self.base_.activation_energy_eV)
        design = np.asarray(design, float).copy()
        design[:, acid_gate.GEM_ACID_COLUMN] = acid_parts["acid_process_new_coordinate"]
        hopping, _ = transport._physics.hopping_coordinate(prepared)
        if not self.base_.gem_.enable_hopping: hopping[:] = 0.
        _, low, _, _ = self.base_.gem_._compute(design, hopping, self.base_.gem_.coef_.copy())
        salt_parts = salt_forward(clean, np.log10(low), acid_parts, self.base_.gem_)
        acid_changed = acid_parts["acid_process_coordinate_changed"] == 1
        salt_changed = salt_parts["salt_active"] & (salt_parts["salt_removed_low_fraction"] > 0)
        use_delta = acid_changed | salt_changed
        delta = salt_parts["prediction_log10_S_cm"]-baseline["prediction_log10_S_cm"]
        domain_delta = salt_parts["domain_correction_log10"]-baseline["domain_correction_log10"]
        for column in (0, 1, 2): result[use_delta, column] += delta[use_delta]
        result[use_delta, 4] += domain_delta[use_delta]
        if not np.array_equal(source[:, 3], result[:, 3]):
            raise AssertionError("Acid/salt composition changed M residual")
        # A separate complete outer-material expert, exactly the original
        # composition convention; it intentionally does not inherit M residual.
        peo_input = peo_inputs(clean)
        peo_use = np.zeros(len(clean), bool)
        if self.peo_.cross_fit_ is not None:
            peo_use = peo_input["active"]
        elif self.peo_.old_fit_ is not None:
            peo_use = peo_input["active"] & (peo_input["family"] == 0)
        if peo_use.any():
            q = self.peo_.predict(clean, result[:, 0])
            result[peo_use, :3] = q[peo_use, None]
            result[peo_use, 3:] = 0.
        changed_scope = use_delta | peo_use
        if not np.array_equal(source[~changed_scope], result[~changed_scope]):
            raise AssertionError("Rows outside admitted original physics changed")
        _values(result, len(clean), "final QUERY values")
        details = {"n_query": len(clean), "prior_fit_n": self.prior_fit_n_,
            "acid_protocol_admitted": int((acid_parts["acid_process_is_admitted"] == 1).sum()),
            "acid_coordinate_changed": int(acid_changed.sum()), "salt_active": int(salt_parts["salt_active"].sum()),
            "salt_positive_extent": int(salt_changed.sum()), "peo_complete_expert": int(peo_use.sum()),
            "combined_changed_scope": int(changed_scope.sum()),
            "actual_total_changed_above_1e_12": int(np.count_nonzero(np.abs(result[:, 0]-source[:, 0]) > 1e-12)),
            "outside_changed": 0, "M_residual_unchanged_outside_complete_PEO_expert":
                bool(np.array_equal(source[~peo_use, 3], result[~peo_use, 3])),
            "baseline_physical_certificates": _certificates(baseline),
            "acid_physical_certificates": _certificates(acid_parts),
            "salt_physical_certificates": _certificates(salt_parts),
            "all_final_decompositions_valid": True, "no_QUERY_y_argument": True,
            "final_ensemble_is_one_GEM_closure": False,
            "cross_PEO_head_fit": self.peo_.cross_fit_ is not None,
            "complete_original_M_retained": False}
        return result, details

    def summary(self):
        if not hasattr(self, "state_"):
            raise RuntimeError("Fit on explicit TRAIN rows first")
        return self.state_
