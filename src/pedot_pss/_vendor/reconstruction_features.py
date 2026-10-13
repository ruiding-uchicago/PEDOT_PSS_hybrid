"""Explicit input reconstruction and low-weight, local GEM-delta augmentation.

This module never uses a specimen ID or a measured electrical response as an input
lookup. Source conditioning is opt-in and labelled calibration. Observed structure
is opt-in. Synthetic labels enforce a local fitted GEM prior; they are not measured
data and cannot establish experimental or out-of-fold accuracy.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import math
import re
import unicodedata

import numpy as np

from .features import build_features, _number, _slug
from .identity_features import enrich_identity_features, normalize_identity_name
from .process_features import (
    CHEMICAL_PATTERNS, CONCENTRATION, NUMBER, _unit, _valid_scalar,
    enrich_process_features,
)
from .quantity_features import quantity_features

FEATURE_VERSION = "reconstruction-inputs-v1"
_OUTCOME = re.compile(r"conductiv|resistiv|resistan|\bsigma\b|σ|seebeck|power.?factor|carrier|\bresults?\b|sample[_ -]?id|row[_ -]?id", re.I)
_VENDORS = ("heraeus", "bayer", "agfa", "sigma-aldrich", "merck", "ossila")
_POST_CHEMICALS = dict(CHEMICAL_PATTERNS, **{
    "trifluoroacetic_acid": r"trifluoroacetic\s+acid|\btfa\b",
    "triflic_acid": r"triflic\s+acid|trifluoromethanesul(?:f|ph)onic\s+acid|\btfoh\b",
    "camphorsulfonic_acid_label": r"camphor\s*sul(?:f|ph)onic\s+acid|\bcsa\b",
    "potassium_hydroxide": r"potassium\s+hydroxide|\bkoh\b",
    "emim_dca": r"\bemim\s*[:/ -]\s*dca\b",
    "emim_tcb": r"\bemim\s*[:/ -]\s*tcb\b",
    "emim_bf4": r"\bemim\s*[:/ -]\s*bf4\b",
    "emim_es": r"\bemim\s*[:/ -]\s*es\b",
    "litfsi": r"\blitfsi\b",
})
# Avoid interpreting the 'acetic acid' substring in trifluoroacetic acid twice.
_POST_CHEMICALS["acetic_acid"] = r"(?<!trifluoro)\bacetic\s+acid|\bch3cooh\b"
_LENGTH = {"nm": 1e-9, "um": 1e-6, "μm": 1e-6, "mm": .001, "cm": .01, "m": 1.0}
_LU = r"(?:nm|um|μm|mm|cm|m)"
_OBSERVED_EXTRA = re.compile(r"thickness|roughness|transmittance|contact_angle|work_function|optical_band_gap|youngs?_modulus|elastic_modulus|tensile_strength|strain_at_break|elongation_at_break|particle_size|fiber_diameter|fibre_diameter|crystallite_size|dielectric_constant|dipole_moment|surface_tension|^density$", re.I)
_CONTROL_EXTRA = frozenset({"blade_gap", "blade_coating_gap", "electrode_spacing", "channel_length", "channel_width", "number_of_layers", "coating_passes", "deposition_cycles", "relative_humidity", "humidity", "pressure", "mixing_speed", "mixing_duration", "uv_exposure_duration", "spin_speed", "spin_duration", "stretch_ratio", "total_draw_ratio"})


def _text(value):
    return unicodedata.normalize("NFKC", value).replace("−", "-") if isinstance(value, str) else ""


def _unit_label(unit):
    """Keep the physical unit, not a value/uncertainty annotation as a category."""
    front = re.split(r"\(|±|\+/-|\b(?:ca|approximately|approx|at)\b", unit, maxsplit=1, flags=re.I)[0].strip()
    if not front or re.match(r"\d", front) or re.search(r"\s\d+(?:\.\d+)?\b", front):
        return None
    if front == "%":
        return "percent"
    return _slug(front.replace("μ", "u")) or None


def _store_unique(out, candidates):
    for key, values in candidates.items():
        unique = set(values)
        out["has_" + key] = float(len(unique) == 1)
        if len(unique) == 1:
            out[key] = next(iter(unique))
        elif unique:
            out["conflicting_" + key] = 1.0


def _post_features(row, out):
    values = defaultdict(list)
    states = defaultdict(list)
    post = _text(row.get("post_treatment"))
    for clause in re.split(r";|→|->", post):
        if _OUTCOME.search(clause):
            continue
        chemicals = [(name, match) for name, pattern in _POST_CHEMICALS.items()
                     for match in re.finditer(pattern, clause, re.I)]
        matches = list(CONCENTRATION.finditer(clause))
        associated = defaultdict(list)
        for match in matches:
            if not _valid_scalar(clause, match):
                continue
            choices = []
            for name, chemical in chemicals:
                between = (clause[match.end():chemical.start()] if chemical.start() >= match.end()
                           else clause[chemical.end():match.start()])
                distance = max(chemical.start() - match.end(), match.start() - chemical.end(), 0)
                words = set(re.findall(r"[a-z]{2,}", between.lower()))
                if distance <= 28 and words <= {"of", "in", "at", "aqueous", "solution", "bath", "concentration", "about", "approximately", "by", "volume", "weight", "mass"} and not re.search(r"[,;]|\b(?:then|and)\b", between, re.I):
                    choices.append((distance, name))
            choices.sort()
            if not choices or (len(choices) > 1 and choices[0][0] == choices[1][0] and choices[0][1] != choices[1][1]):
                continue
            unit, factor = _unit(match.group("unit"))
            value = float(match.group("value")) * factor
            if math.isfinite(value):
                name = choices[0][1]
                associated[name].append(value)
                values[f"recon_post_{name}_solution_{unit}"].append(value)
        for name, match in chemicals:
            before = clause[max(0, match.start() - 24):match.start()]
            negated = bool(re.search(r"\b(?:no|without|before|not)\s+(?:[a-z-]+\s+){0,2}$", before, re.I))
            concentrations = associated.get(name, [])
            present = not negated and not (concentrations and all(value == 0 for value in concentrations))
            states[f"recon_post_{name}_present"].append(float(present))
    for key, candidates in states.items():
        out[key] = max(candidates)
    _store_unique(out, values)


def _dimension_features(row, out, include_observed):
    candidates = defaultdict(list)
    for field in ("substrate", "specimen_geometry", "measurement_method", "process_sequence"):
        text = _text(row.get(field)).casefold()
        for clause in re.split(r";|→|->", text):
            if _OUTCOME.search(clause):
                continue
            if field in {"substrate", "specimen_geometry"}:
                pair = re.search(rf"(?P<x>{NUMBER})\s*(?P<ux>{_LU})?\s*[x×]\s*(?P<y>{NUMBER})\s*(?P<uy>{_LU})\b", clause)
                if pair and _valid_scalar(clause, pair):
                    ux, uy = pair.group("ux") or pair.group("uy"), pair.group("uy")
                    candidates[f"recon_{field}_size_x_m"].append(float(pair.group("x")) * _LENGTH[ux])
                    candidates[f"recon_{field}_size_y_m"].append(float(pair.group("y")) * _LENGTH[uy])
            roles = {"blade_gap": r"(?:blade\s+)?gap", "electrode_spacing": r"(?:electrode|probe|contact)\s+(?:spacing|separation)", "channel_length": r"channel\s+length", "channel_width": r"channel\s+width"}
            if include_observed:
                roles.update({"diameter": r"(?:via\s+|fiber\s+|fibre\s+)?diameter", "thickness": r"(?:film\s+)?thickness"})
            for role, label in roles.items():
                patterns = (rf"(?:{label})\s*(?:of|=|:)?\s*(?P<v>{NUMBER})\s*(?P<u>{_LU})\b", rf"(?P<v>{NUMBER})\s*(?P<u>{_LU})\s*(?:{label})\b")
                for pattern in patterns:
                    for match in re.finditer(pattern, clause):
                        if _valid_scalar(clause, match):
                            candidates[f"recon_{field}_{role}_m"].append(float(match.group("v")) * _LENGTH[match.group("u")])
    _store_unique(out, candidates)


def build_reconstruction_features(original_row, *, include_observed=False, include_context=False):
    """Compose richer input-only features. Context means disclosed paper calibration.

    Identical complete features with differing measured targets cannot be mapped
    exactly by any deterministic predictor; use collision_diagnostics to quantify
    that limit rather than inserting a row ID or measured response as an input.
    """
    row = original_row
    out = build_features(row, include_observed=include_observed, include_context=False)
    out = enrich_identity_features(row, out)
    out = enrich_process_features(row, out)
    out.update(quantity_features(row))
    grade_raw = _text(row.get("polymer_grade"))
    # 'Sigma-Aldrich' is a named supplier, not an electrical sigma response.
    grade = normalize_identity_name(re.sub(r"sigma[- ]aldrich", "sigmaaldrich", grade_raw, flags=re.I))
    if grade:
        out["recon_material_grade_label"] = grade
        for vendor in _VENDORS:
            out["recon_vendor_" + _slug(vendor)] = float(vendor in grade_raw.casefold())
    _post_features(row, out)
    _dimension_features(row, out, include_observed)
    candidates = defaultdict(list)
    for item in row.get("extra_scalar_features") or []:
        if not isinstance(item, Mapping):
            continue
        name = _slug(item.get("name"))
        if _OUTCOME.search(name) or len(name) > 100:
            continue
        if name not in _CONTROL_EXTRA and not (include_observed and _OBSERVED_EXTRA.search(name)):
            continue
        value, unit = _number(item.get("value")), _text(item.get("unit")).strip()
        if value is None or not unit or len(unit) > 40 or _OUTCOME.search(unit):
            continue
        unit_label = _unit_label(unit)
        if unit_label:
            candidates[f"recon_extra_{name}__{unit_label}"].append(value)
    _store_unique(out, candidates)
    if include_context:
        paper = row.get("paper_id")
        if isinstance(paper, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", paper) and not re.search(r"[-_:]r\d", paper, re.I):
            out["recon_calibration_paper"] = paper.casefold()
        batch = _text(row.get("batch")).casefold()
        out["recon_calibration_platform"] = "polybot" if "polybot" in batch else "literature"
        out["recon_source_calibration_enabled"] = 1.0
    return out


def collision_diagnostics(features, y):
    """Exact-X deterministic reconstruction ceiling; an audit, never a model feature."""
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or len(target) != len(features) or not np.isfinite(target).all():
        raise ValueError("features/y must align and y must be finite")
    groups = defaultdict(list)
    for i, row in enumerate(features):
        groups[json.dumps(dict(row), sort_keys=True, ensure_ascii=False, allow_nan=False)].append(i)
    sse = 0.0
    ambiguous = []
    for indices in groups.values():
        values = target[indices]
        error = float(np.sum((values - values.mean()) ** 2))
        sse += error
        if len(indices) > 1 and error > 1e-20:
            ambiguous.append({"row_positions": indices, "n": len(indices), "target_log10_range": [float(values.min()), float(values.max())], "unavoidable_sse": error})
    sst = float(np.sum((target - target.mean()) ** 2)) if len(target) else 0.0
    return {"n_rows": len(target), "distinct_feature_rows": len(groups), "ambiguous_groups": ambiguous,
            "irreducible_sse_log10": sse, "deterministic_reconstruction_r2_ceiling": 1.0 - sse / sst if sst > 0 else None,
            "interpretation": "Exact input collisions only; no claim about generalization, measurement error, or achievable finite-model fit."}


def augment_training(features, y, physics_predictor, *, n_copies=1,
                     relative_jitter=.01, random_state=20261009, sample_weight=None,
                     synthetic_weight=.25):
    """Return features, log10 targets, weights, metadata; originals come first.

    Default augmentation perturbs only a reported PSS:PEDOT mass ratio, an actual
    GEM coordinate with no repeated Polybot control encoding. No missing ratio is
    invented. Ratios with a second numeric ratio encoding are conservatively
    skipped. No row, dose basis, grade, phase label or observed structure is jittered.
    Requires a fitted real GEM predictor exposing predict() and components().
    Synthetic y = measured parent y + GEM(new X) - GEM(parent X). This local prior
    preserves the parent's fitted residual but is not independent measured evidence.
    """
    originals = [dict(row) for row in features]
    target = np.asarray(y, dtype=float)
    weights = np.ones(len(originals)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if target.ndim != 1 or len(target) != len(originals) or not np.isfinite(target).all():
        raise ValueError("features/y must align with finite log10 targets")
    if weights.shape != target.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid sample_weight")
    if not isinstance(n_copies, int) or n_copies < 0 or not 0 <= relative_jitter <= .1 or not 0 <= synthetic_weight <= .25:
        raise ValueError("Require n_copies>=0, local jitter<=0.1, synthetic_weight<=0.25")
    if not callable(getattr(physics_predictor, "predict", None)) or not callable(getattr(physics_predictor, "components", None)):
        raise TypeError("Pass the actual fitted GEM physics component, not a residual learner")
    rng = np.random.default_rng(random_state)
    augmented = [dict(row, augmentation_is_synthetic=0.0) for row in originals]
    synthetic, parents, changes = [], [], []
    key = "reported_pss_pedot_mass_ratio"
    skipped = 0
    for i, row in enumerate(originals):
        value = _number(row.get(key))
        duplicates = [name for name in row if name != key and not name.startswith(("has_", "__present__")) and "ratio" in name.lower() and "pss" in name.lower() and isinstance(row[name], (int, float))]
        if value is None or value <= 0 or value > 100 or duplicates:
            skipped += 1
            continue
        for _ in range(n_copies):
            changed = value * (1.0 + rng.uniform(-relative_jitter, relative_jitter))
            if changed == value or not 0 < changed <= 100:
                continue
            candidate = dict(row)
            candidate[key] = changed
            synthetic.append(candidate); parents.append(i)
            changes.append({"parent_index": i, "coordinate": key, "old": value, "new": changed})
    if synthetic:
        baseline = np.asarray(physics_predictor.predict(originals), dtype=float)
        prediction = np.asarray(physics_predictor.predict(synthetic), dtype=float)
        components = physics_predictor.components(synthetic)
        residual = np.asarray(components.get("gem_equation_residual"), dtype=float)
        if baseline.shape != target.shape or prediction.shape != (len(synthetic),) or residual.shape != prediction.shape or not np.isfinite(baseline).all() or not np.isfinite(prediction).all() or not np.isfinite(residual).all() or np.any(np.abs(residual) > 1e-8):
            raise ValueError("Nonfinite/invalid GEM prediction or implicit-equation check")
        delta = prediction - baseline[np.asarray(parents)]
        synthetic_y = target[np.asarray(parents)] + delta
        for row, record, difference in zip(synthetic, changes, delta):
            augmented.append(dict(row, augmentation_is_synthetic=1.0))
            record["gem_delta_log10_S_cm"] = float(difference)
        augmented_y = np.concatenate((target.copy(), synthetic_y))
        augmented_weights = np.concatenate((weights.copy(), np.minimum(weights[np.asarray(parents)] * synthetic_weight, .25)))
    else:
        augmented_y, augmented_weights = target.copy(), weights.copy()
    metadata = {"n_original": len(originals), "n_synthetic": len(synthetic), "n_skipped_without_unique_reported_ratio": skipped,
                "label_rule": "parent measured log10 y + fitted real GEM(new X) - fitted real GEM(parent X)",
                "synthetic_weight_max": .25, "jitter_coordinate": key, "relative_jitter": relative_jitter,
                "random_state": random_state, "records": changes, "original_table_modified": False,
                "scope_warning": "Synthetic local-prior training rows, not measurements or held-out validation."}
    return augmented, augmented_y, augmented_weights, metadata
