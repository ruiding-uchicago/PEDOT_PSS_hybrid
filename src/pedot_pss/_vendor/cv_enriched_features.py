"""Stateless, input-only additions for retrospective state-model CV.

Return an *incremental* dictionary: callers update their existing feature view.
Nothing is fitted and no file, target, electrical observation, or conductivity
uncertainty is read. No specimen/source identity is emitted; state-ID suffixes
only verify the first-dry observation stage. The optional observed features make this a
post-characterization state model, not a recipe-only prospective predictor.

In particular, first-dry coverage SD is the SD of optical image coverage (%),
not the SD of conductivity.  It describes spatial film heterogeneity before
annealing/post-treatment.  Its sample count/pooling are unknown: no SEM, CI,
replicate count, or final-state connectivity is inferred.  Features combining
coverage mean/SD are descriptive transforms, not measured percolation factors.

Reported Polybot additive vol% retain the unresolved stock-versus-final-mixture
denominator.  Coating stage setpoints are not measured film temperatures, and
inverse speeds are per-length proxies, not invented coating durations.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math
import re

from .features import _number, _slug
from .polybot_features import polybot_features

ENRICHED_FEATURE_VERSION = "cv-input-structure-v1"
_FORBIDDEN = re.compile(
    r"conductiv|resist|sigma|seebeck|power.?factor|carrier|mobility|"
    r"activation_energy|(?:^|_)t0(?:_|$)|vrh|(?:^|_)id(?:_|$)|"
    r"sample|specimen|replicat|electrical|uncertainty", re.I)

# These aliases name physical characterizations or explicitly reported process
# controls.  Arbitrary extra-field names and electrical proxies are not copied.
_ALIASES = {
    "rms_roughness": ("roughness_rms_nm", "length_nm"),
    "root_mean_square_roughness": ("roughness_rms_nm", "length_nm"),
    "surface_roughness_rms": ("roughness_rms_nm", "length_nm"),
    "surface_rms_roughness": ("roughness_rms_nm", "length_nm"),
    "rms_surface_roughness": ("roughness_rms_nm", "length_nm"),
    "afm_rq": ("roughness_rms_nm", "length_nm"),
    "afm_rms": ("roughness_rms_nm", "length_nm"),
    "afm_rms_roughness": ("roughness_rms_nm", "length_nm"),
    "rq": ("roughness_rms_nm", "length_nm"),
    "surface_roughness_ra": ("roughness_ra_nm", "length_nm"),
    "ra": ("roughness_ra_nm", "length_nm"),
    "surface_roughness": ("roughness_unspecified_nm", "length_nm"),
    "roughness": ("roughness_unspecified_nm", "length_nm"),
    "thickness": ("thickness_nm", "length_nm"),
    "film_thickness": ("thickness_nm", "length_nm"),
    "film_thickness_reported": ("thickness_nm", "length_nm"),
    "thickness_from_profilometry": ("profilometry_thickness_nm", "length_nm"),
    "etched_film_thickness": ("etched_thickness_nm", "length_nm"),
    "particle_size": ("particle_size_nm", "length_nm"),
    "crystallite_size": ("crystallite_size_nm", "length_nm"),
    "fiber_diameter": ("fiber_diameter_nm", "length_nm"),
    "fibre_diameter": ("fiber_diameter_nm", "length_nm"),
    "transmittance_at_550_nm": ("transmittance_550_fraction", "fraction"),
    "transmittance_550nm": ("transmittance_550_fraction", "fraction"),
    "transmittance_550_nm": ("transmittance_550_fraction", "fraction"),
    "visible_transmittance": ("transmittance_visible_fraction", "fraction"),
    "haze": ("haze_fraction", "fraction"),
    "absorbance_430": ("absorbance_430", "absorbance"),
    "contact_angle": ("contact_angle_deg", "angle"),
    "water_contact_angle": ("water_contact_angle_deg", "angle"),
    "youngs_modulus": ("youngs_modulus_Pa", "pressure"),
    "young_modulus": ("youngs_modulus_Pa", "pressure"),
    "elastic_modulus": ("elastic_modulus_Pa", "pressure"),
    "tensile_modulus": ("tensile_modulus_Pa", "pressure"),
    "tensile_strength": ("tensile_strength_Pa", "pressure"),
    "ultimate_stress": ("ultimate_stress_Pa", "pressure"),
    "drying_tensile_stress": ("drying_tensile_stress_Pa", "pressure"),
    "strain_at_break": ("strain_at_break_fraction", "strain"),
    "elongation_at_break": ("elongation_at_break_fraction", "strain"),
    "stretchability": ("stretchability_fraction", "strain"),
    "crack_onset_strain": ("crack_onset_strain_fraction", "strain"),
    "bending_strain": ("bending_strain_fraction", "strain"),
    "bending_cycles": ("bending_cycles", "count"),
    "zeta_potential": ("zeta_potential_mV", "millivolt"),
    "neutral_state": ("optical_neutral_state_fraction", "fraction"),
    "polaron_state": ("optical_polaron_state_fraction", "fraction"),
    "bipolaron_state": ("optical_bipolaron_state_fraction", "fraction"),
}


def _put(out, name, value):
    value = _number(value)
    out["cv_has_" + name] = float(value is not None)
    if value is not None:
        out["cv_" + name] = float(value)


def _unit(unit):
    return re.sub(r"\s+", "", str(unit or "").lower().replace("μ", "u").replace("µ", "u"))


def _convert(value, unit, kind):
    value = _number(value)
    if value is None:
        return None
    unit = _unit(unit)
    if kind == "length_nm":
        factor = {"nm": 1., "um": 1000., "mm": 1e6, "a": .1, "å": .1}.get(unit)
    elif kind in {"fraction", "strain"}:
        factor = {"%": .01, "percent": .01, "fraction": 1., "dimensionless": 1.}.get(unit)
    elif kind == "pressure":
        factor = {"pa": 1., "kpa": 1e3, "mpa": 1e6, "gpa": 1e9}.get(unit)
    elif kind == "angle":
        factor = {"degree": 1., "degrees": 1., "deg": 1., "°": 1.}.get(unit)
    elif kind == "absorbance":
        factor = {"a.u.": 1., "a.u": 1., "au": 1., "dimensionless": 1.}.get(unit)
    elif kind == "count":
        factor = {"cycles": 1., "cycle": 1., "count": 1.}.get(unit)
    elif kind == "millivolt":
        factor = {"mv": 1., "v": 1000.}.get(unit)
    else:
        factor = None
    if factor is None:
        return None
    converted = value * factor
    if kind == "fraction" and not 0 <= converted <= 1:
        return None
    if kind not in {"millivolt"} and converted < 0:
        return None
    return converted


def _reported_percent(quantity):
    if not isinstance(quantity, Mapping) or quantity.get("status") != "reported":
        return None
    value = quantity.get("value")
    if not isinstance(value, Mapping) or value.get("kind") != "point":
        return None
    number = _number(value.get("value"))
    if _unit(quantity.get("unit")) not in {"%", "percent"}:
        return None
    return number if number is not None and 0 <= number <= 100 else None


def _first_dry_coverage_observation(row):
    # Exactly one structural property is allowed.  We never iterate arbitrary
    # measurement_associations (which also contains conductivity/its SD).
    associations = row.get("measurement_associations")
    if isinstance(associations, Mapping):
        candidate = associations.get("coverage")
        if isinstance(candidate, Mapping) and str(candidate.get("state_id", "")).endswith(":first-dry"):
            return candidate
    # Native container support uses the same exact property and state check.
    candidates = [item for item in (row.get("observations") or [])
                  if isinstance(item, Mapping) and item.get("property") == "coverage"
                  and str(item.get("state_id", "")).endswith(":first-dry")]
    return candidates[0] if len(candidates) == 1 else None


def _coverage_features(row, out):
    observation = _first_dry_coverage_observation(row)
    mean = _number(row.get("coverage_percent"))
    if mean is not None and not 0 <= mean <= 100:
        mean = None
    observed_mean = _reported_percent(observation.get("result")) if observation else None
    if mean is None:
        mean = observed_mean
    elif observed_mean is not None and not math.isclose(mean, observed_mean, rel_tol=1e-9, abs_tol=1e-7):
        # Conflicting source views must not be silently reconciled.
        out["cv_conflicting_observed_coverage_mean"] = 1.0
        mean = None
    sd = None
    if observation:
        uncertainty = observation.get("uncertainty")
        if (isinstance(uncertainty, Mapping) and uncertainty.get("kind") == "SD"
                and uncertainty.get("representation") == "absolute"):
            sd = _reported_percent(uncertainty.get("quantity"))
    _put(out, "observed_coverage_fraction", None if mean is None else mean / 100.)
    _put(out, "observed_first_dry_coverage_sd_percent", sd)
    if mean is not None:
        fraction = mean / 100.
        _put(out, "observed_uncovered_fraction", 1. - fraction)
        _put(out, "observed_coverage_logit_smoothed", math.log((fraction + .005) / (1.005 - fraction)))
        _put(out, "observed_coverage_zero", float(mean == 0))
        _put(out, "observed_coverage_full", float(mean == 100))
    if sd is not None:
        _put(out, "observed_first_dry_coverage_sd_fraction", sd / 100.)
        _put(out, "observed_first_dry_coverage_variance_fraction2", (sd / 100.) ** 2)
        _put(out, "observed_first_dry_coverage_log1p_sd_percent", math.log1p(sd))
    if mean is not None and sd is not None:
        # Clipped mean +/- SD are descriptive spatial summaries, NOT confidence
        # bounds or a claimed area fraction of electrically connected material.
        _put(out, "observed_first_dry_coverage_mean_minus_sd_fraction", max(0., mean - sd) / 100.)
        _put(out, "observed_first_dry_coverage_mean_plus_sd_fraction", min(100., mean + sd) / 100.)
        _put(out, "observed_first_dry_coverage_spatial_cv", sd / mean if mean > 0 else None)
        _put(out, "observed_first_dry_coverage_mean_x_sd_fraction2", mean * sd / 1e4)


def _observed_extras(row, out):
    values = defaultdict(list)
    thickness = _number(row.get("thickness_nm"))
    if thickness is not None and thickness > 0:
        values["thickness_nm"].append(thickness)
    for item in row.get("extra_scalar_features") or []:
        if not isinstance(item, Mapping):
            continue
        name = _slug(item.get("name"))
        if _FORBIDDEN.search(name):
            continue
        alias = _ALIASES.get(name)
        # A stated optical wavelength is retained; generic/visible transmittance
        # are not silently interpreted as a 550 nm measurement.
        if name == "transmittance":
            unit = _unit(item.get("unit"))
            if unit in {"%at550nm", "%at550_nm"}:
                alias = ("transmittance_550_fraction", "fraction")
                value = _convert(item.get("value"), "%", "fraction")
            else:
                alias = ("transmittance_unspecified_fraction", "fraction")
                value = _convert(item.get("value"), item.get("unit"), "fraction")
        elif alias:
            value = _convert(item.get("value"), item.get("unit"), alias[1])
        else:
            continue
        if value is not None:
            values[alias[0]].append(value)
    for name, candidates in values.items():
        if max(candidates) - min(candidates) > 1e-8 * max(1., max(abs(v) for v in candidates)):
            out["cv_has_observed_" + name] = 0.
            out["cv_conflicting_observed_" + name] = 1.
            continue
        value = candidates[0]
        _put(out, "observed_" + name, value)
        if value >= 0:
            _put(out, "observed_log1p_" + name, math.log1p(value))
        if name.endswith("transmittance_550_fraction") and value > 0:
            _put(out, "observed_optical_depth_550", -math.log(value))
    # Dimensionless roughness/thickness is structural only, never R_sheet*t.
    thickness = out.get("cv_observed_thickness_nm")
    roughness = out.get("cv_observed_roughness_rms_nm")
    if thickness is not None and thickness > 0 and roughness is not None:
        _put(out, "observed_rms_roughness_over_thickness", roughness / thickness)


def _recipe_transforms(row, out):
    for field in ("stock_solids_wt_percent", "dry_duration_min", "anneal_duration_min",
                  "post_treatment_duration_min"):
        value = _number(row.get(field))
        if value is not None and value >= 0:
            _put(out, "recipe_log1p_" + field, math.log1p(value))
    for field in ("dry_temperature_C", "anneal_temperature_C", "post_treatment_temperature_C"):
        value = _number(row.get(field))
        if value is not None and value > -273.15:
            _put(out, "recipe_1000_over_" + field.replace("_C", "_K"), 1000. / (value + 273.15))
    pb = polybot_features(row)
    if not pb.get("pb_controls_complete"):
        return
    # All values below are functions of the existing physical control whitelist.
    dmso = pb["pb_dmso_volpct_reported"]
    eg = pb["pb_eg_volpct_reported"]
    total = dmso + eg
    _put(out, "pb_reported_additive_volpct_product", dmso * eg)
    _put(out, "pb_reported_additive_volpct_sum_log1p", math.log1p(total))
    _put(out, "pb_dmso_reported_volpct_saturation", dmso / (1. + dmso))
    _put(out, "pb_eg_reported_volpct_saturation", eg / (1. + eg))
    _put(out, "pb_dmso_fraction_of_reported_additives", dmso / total if total > 0 else None)
    first_speed, post_speed = pb["pb_first_coating_speed_mm_s"], pb["pb_post_speed_mm_s"]
    first_temp = pb["pb_first_coating_stage_temperature_C"]
    post_temp = pb["pb_post_stage_temperature_C"]
    for stage, speed, temp in (("first", first_speed, first_temp), ("post", post_speed, post_temp)):
        if speed > 0:
            _put(out, "pb_" + stage + "_log_speed_mm_s", math.log(speed))
            _put(out, "pb_" + stage + "_inverse_speed_s_mm", 1. / speed)
            _put(out, "pb_" + stage + "_temp_C_over_speed_mm_s", temp / speed)
            gap = pb.get("pb_blade_gap_um")
            if stage == "first" and gap is not None and gap > 0:
                _put(out, "pb_first_nominal_speed_over_gap_s_inv", 1000. * speed / gap)
        if temp > -273.15:
            _put(out, "pb_" + stage + "_1000_over_stage_temperature_K", 1000. / (temp + 273.15))
    _put(out, "pb_stage_temperature_post_minus_first_C", post_temp - first_temp)
    _put(out, "pb_log_post_over_first_speed", math.log(post_speed / first_speed)
         if first_speed > 0 and post_speed > 0 else None)
    _put(out, "pb_first_temperature_x_reported_additive_sum", first_temp * total)
    _put(out, "pb_first_inverse_speed_x_reported_additive_sum", total / first_speed if first_speed > 0 else None)
    fractions = [pb["pb_post_" + solvent + "_volpct"] / 100. for solvent in ("methanol", "ethanol", "water")]
    _put(out, "pb_post_composition_entropy", -sum(f * math.log(f) for f in fractions if f > 0))
    _put(out, "pb_post_methanol_ethanol_fraction_product", fractions[0] * fractions[1])
    _put(out, "pb_post_water_x_stage_temperature_C", fractions[2] * post_temp)
    coverage = out.get("cv_observed_coverage_fraction")
    sd = out.get("cv_observed_first_dry_coverage_sd_fraction")
    if coverage is not None:
        _put(out, "pb_observed_coverage_x_post_water_fraction", coverage * fractions[2])
        _put(out, "pb_observed_uncovered_x_post_speed_mm_s", (1. - coverage) * post_speed)
    if sd is not None:
        _put(out, "pb_observed_coverage_sd_x_post_speed_mm_s", sd * post_speed)


def cv_enriched_features(original_row: Mapping) -> dict[str, float]:
    """Return only additional ``cv_*`` features; no training or side effects.

    Usage: ``features.update(cv_enriched_features(example["original_row"]))``.
    This routine intentionally enables observed structure.  Do not silently use
    it in a profile described as recipe-only or as an untouched prospective test.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("original_row must be a mapping")
    out = {}
    _coverage_features(original_row, out)
    _observed_extras(original_row, out)
    _recipe_transforms(original_row, out)
    return out


# Explicit opt-in, higher-risk metadata profile. The strict API above is frozen.
LABEL_CONDITION_FEATURE_VERSION = "cv-explicit-label-conditions-v1"
_LC_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LC_SOLVENTS = {
    "dmf": r"dmf|dimethylformamide", "gbl": r"gbl|gamma.butyrolactone",
    "acetone": r"acetone", "acetonitrile": r"acn|acetonitrile",
    "methanol": r"methanol|meoh", "ethanol": r"ethanol|etoh",
    "ipa": r"ipa|isopropanol", "dmso": r"dmso|dimethyl\s+sulfoxide",
    "methoxyethanol": r"2-me|2-methoxyethanol|methoxyethanol",
    "water": r"water|h2o",
}
_LC_DOPANTS = {
    "dae": r"dae", "litfsi": r"li\s*[: -]?\s*tfsi|litfsi",
    "laa": r"laa|l-ascorbic\s+acid|ascorbic\s+acid",
    "ho_bsa": r"ho-bsa", "cl_bsa": r"cl-bsa", "ebsa": r"ebsa",
    "bsa": r"(?<![a-z-])bsa", "tsa": r"tsa", "msa": r"msa",
    "nsa": r"nsa", "csa": r"csa", "mai": r"mai",
    "hno3": r"hno3|nitric\s+acid", "h3po4": r"h3po4|phosphoric\s+acid",
    "h2so4": r"h2so4|sulfuric\s+acid", "hydrazine": r"hz|hydrazine",
}


def _lc_text(value):
    import unicodedata
    return unicodedata.normalize("NFKC", str(value or "")).lower().replace("−", "-").replace("µ", "u").replace("μ", "u")


def _lc_store_unique(out, name, values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return
    if max(values) - min(values) > 1e-8 * max(1., max(abs(v) for v in values)):
        out["cv_label_conflicting_" + name] = 1.
    else:
        out["cv_label_" + name] = values[0]


def cv_label_condition_features(original_row: Mapping) -> dict[str, float | str]:
    """Opt-in additions recovered from explicit input/label conditions.

    This higher-risk retrospective profile reads sample_label and series_label
    ONLY through a closed semantic parser for chemistry, geometry, storage and
    processing. No raw labels, paper/sample IDs, digitized point/square ordinal,
    IV-shape label, percentile/median/rank, optimal/outlier status or response
    quantity is emitted. ``feature_evidence`` and association_notes are never
    read: these often contain the answer. Call separately from the frozen strict
    ``cv_enriched_features`` and disclose that metadata conditions are enabled.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("original_row must be a mapping")
    row = original_row
    label = _lc_text(row.get("sample_label"))
    series = _lc_text(row.get("series_label"))
    process = _lc_text(row.get("process_sequence"))
    post = _lc_text(row.get("post_treatment"))
    grade = _lc_text(row.get("polymer_grade"))
    geometry = _lc_text(row.get("specimen_geometry"))
    substrate = _lc_text(row.get("substrate"))
    method = _lc_text(row.get("measurement_method"))
    additive_names = row.get("additive_names") or []
    names = " ".join(_lc_text(x) for x in additive_names) if isinstance(additive_names, list) else ""
    # Labels are not exported; only the closed matches below can survive.
    inputs = "; ".join((label, series, process, post, grade, geometry, substrate, names))
    out: dict[str, float | str] = {}
    for name, pattern in _LC_DOPANTS.items():
        if re.search(r"\b(?:" + pattern + r")\b", inputs):
            out["cv_label_reported_chemical_" + name] = 1.
    for name, pattern in _LC_SOLVENTS.items():
        if re.search(r"\b(?:" + pattern + r")\b", inputs):
            out["cv_label_reported_solvent_" + name] = 1.
    if re.search(r"\bpure\s+dae\b", "; ".join((label, geometry, process))):
        out["cv_label_material_family"] = "pure_dae_control"
        out["cv_label_material_pure_dae"] = 1.
    elif re.search(r"\bpssna\b|poly\(sodium.{0,12}styrene\s+sulfonate", "; ".join((label, grade, process))):
        out["cv_label_material_family"] = "sodium_pss_control"
    elif re.search(r"pedot\s*[-:]\s*tos\b|pedot.{0,12}tosylate", "; ".join((label, grade, process))):
        out["cv_label_material_family"] = "pedot_tos"
    elif re.search(r"pedot\s*[-:]\s*pss\b", "; ".join((label, grade, process))):
        out["cv_label_material_family"] = "pedot_pss"
    if re.search(r"\bhemin\b", inputs):
        out["cv_label_synthesis_oxidant_hemin"] = 1.
    if re.search(r"\bfecl3\b", inputs):
        out["cv_label_synthesis_oxidant_fecl3"] = 1.
    if re.search(r"\bhcg\b", grade + "; " + label):
        out["cv_label_material_commercial_hcg"] = 1.
    for material_grade, pattern in {"icp1020": r"\bicp\s*1020\b", "ph1000": r"\bph\s*1000\b",
                                    "ai4083": r"\b(?:ai|al)\s*4083\b"}.items():
        if re.search(pattern, grade + "; " + label):
            out["cv_label_material_grade_" + material_grade] = 1.
    for name, pattern in {"zno_underlayer": r"\bzno\b", "graphene": r"\bgraphene\b",
                          "polyimide": r"polyimide|kapton"}.items():
        if re.search(pattern, inputs):
            out["cv_label_material_" + name] = 1.
    # A shared grade description may mention PEO even on a pristine control.
    # Require explicit row processing/label evidence before calling it a blend.
    if re.search(r"\bpeo\b", process + "; " + label):
        out["cv_label_material_peo_blend"] = 1.
    # Explicit UV state has priority over UV-name presence alone.
    illumination = "; ".join((label, process, method))
    uv_off = bool(re.search(r"\buv(?:\s+light)?(?:\s+state)?\s+off\b|\bwithout\s+uv\b", illumination))
    uv_on = bool(re.search(r"\buv(?:\s+light)?(?:\s+state)?\s+on\b|\b\d+\s*nm\s+uv\s+on\b|under\s+uv\s+illumination|uv-induced\s+condition", illumination))
    if uv_off != uv_on:
        out["cv_label_uv_on"] = float(uv_on)
    elif uv_off:
        out["cv_label_conflicting_uv_state"] = 1.
    if uv_on:
        for pattern, name in ((rf"({_LC_NUMBER})\s*nm\s+uv", "uv_wavelength_nm"),
                              (rf"({_LC_NUMBER})\s*w\s*cm\s*(?:\^\s*)?-2", "uv_irradiance_W_cm2")):
            _lc_store_unique(out, name, [float(m.group(1)) for m in re.finditer(pattern, illumination)])
    # Amounts are bound to a named chemical and a stated unit. We do not resolve
    # an ambiguous denominator or interpret a compound code as a molecular ID.
    quantity_texts = [label, post]
    amounts = row.get("additive_amounts_raw") or []
    for chemical, pattern in _LC_DOPANTS.items():
        bound = []
        if isinstance(additive_names, list) and isinstance(amounts, list) and len(additive_names) == len(amounts):
            bound = [_lc_text(a) for n, a in zip(additive_names, amounts)
                     if re.search(r"\b(?:" + pattern + r")\b", _lc_text(n))]
        for unit_name, unit_pattern in (("wtpct", r"wt\.?\s*%"), ("volpct", r"vol\.?\s*%"),
                                        ("pct_basis_unresolved", r"%"), ("mol_L", r"M\b")):
            values = []
            flags = 0 if unit_name == "mol_L" else re.I
            # Lowercasing preserves the unit token for weight/volume; molarity
            # accepts a lowercase m only here because the exact raw suffix M
            # is verified using the original amount/label strings below.
            if unit_name == "mol_L":
                raw_texts = [str(row.get("sample_label") or ""), str(row.get("post_treatment") or "")]
                raw_bound = [str(a) for n, a in zip(additive_names, amounts)
                             if re.search(r"\b(?:" + pattern + r")\b", _lc_text(n))] if isinstance(additive_names, list) and isinstance(amounts, list) and len(additive_names) == len(amounts) else []
                for text in raw_bound:
                    matches = re.findall(rf"({_LC_NUMBER})\s*M\b", text)
                    if len(matches) == 1:
                        values.append(float(matches[0]))
                for text in raw_texts:
                    for m in re.finditer(rf"({_LC_NUMBER})\s*M\s+(?:of\s+)?(?:{pattern})\b", text, re.I):
                        if re.search(r"\d\s*M\b", m.group(0)):
                            values.append(float(m.group(1)))
            else:
                for text in bound:
                    matches = re.findall(rf"({_LC_NUMBER})\s*{unit_pattern}", text, flags)
                    if len(matches) == 1:
                        values.append(float(matches[0]))
                for text in quantity_texts:
                    for m in re.finditer(rf"({_LC_NUMBER})\s*{unit_pattern}\s*(?:of\s+)?(?:{pattern})\b", text, re.I):
                        values.append(float(m.group(1)))
            if chemical == "dae" and unit_name == "wtpct":
                values.extend(float(m.group(1)) for m in re.finditer(rf"\b(?:x\s*=)\s*({_LC_NUMBER})\s*wt\.?\s*%", label) if "dae" in label)
            _lc_store_unique(out, chemical + "_reported_" + unit_name, values)
        if re.search(r"\b(?:no|without)\s+(?:" + pattern + r")\b", label + "; " + process + "; " + post):
            out["cv_label_explicitly_absent_chemical_" + chemical] = 1.
    # Geometry: diameter and layer count are meaningful quantities. The trailing
    # square/point ordinal and percentile labels are deliberately not inspected.
    diameters = []
    for m in re.finditer(rf"\bvia(?:\s+diameter)?\s*({_LC_NUMBER})\s*(nm|um|mm)\b", label + "; " + geometry):
        diameters.append(float(m.group(1)) * {"nm": 1., "um": 1e3, "mm": 1e6}[m.group(2)])
    _lc_store_unique(out, "via_diameter_nm", diameters)
    diameter = out.get("cv_label_via_diameter_nm")
    if isinstance(diameter, float) and diameter > 0:
        out["cv_label_log_via_diameter_nm"] = math.log(diameter)
        out["cv_label_via_area_m2"] = math.pi * (diameter * 1e-9 / 2.) ** 2
    layers = [float(m.group(1)) for m in re.finditer(r"\b(\d+)\s*x\s*(?:\[\s*)?pedot", label + "; " + process)]
    layers.extend(float(m.group(1)) for m in re.finditer(r"\b(\d+)[ -]layer\s+pedot", process))
    if re.search(r"\bone\s+layer\b", process):
        layers.append(1.)
    _lc_store_unique(out, "pedot_layer_count", layers)
    if re.search(r"trilayer\s+graphene|graphene\s+trilayer", inputs):
        out["cv_label_graphene_layer_count"] = 3.
    elif re.search(r"graphene\s+monolayer|monolayer\s+graphene", inputs):
        out["cv_label_graphene_layer_count"] = 1.
    if re.search(r"\bin-plane\b", label + "; " + geometry):
        out["cv_label_measurement_direction_in_plane"] = 1.
    elif re.search(r"out-of-plane|vertical\s+junction", label + "; " + geometry):
        out["cv_label_measurement_direction_in_plane"] = 0.
    # Storage durations require a time unit and a storage/age context, or the
    # explicit after-fabrication label. Dates/row numbers cannot match this.
    months = [float(m.group(1)) for m in re.finditer(rf"({_LC_NUMBER})[ -]+months?\b", label)]
    if re.search(r"after[ -]fabrication", label):
        months.append(0.)
    _lc_store_unique(out, "storage_duration_months", months)
    # Repeated, semantic fabrication/water codes are admitted, not arbitrary
    # alphanumeric sample identifiers. Their abbreviations remain uninterpreted.
    code = re.match(r"^(?:dc(?:-tc)?|extc)-(dm|m)-", label)
    if code:
        out["cv_label_reported_water_preparation_code"] = code.group(1)
    if "extrusion" in process or re.match(r"^extc-", label):
        out["cv_label_fabrication_family"] = "extrusion_thermocompression"
    elif "thermocompress" in process or re.match(r"^dc-tc-", label):
        out["cv_label_fabrication_family"] = "dropcast_thermocompression"
    elif re.match(r"^dc-", label):
        out["cv_label_fabrication_family"] = "dropcast"
    if "peo" in grade and (re.match(r"^(?:dc|extc)-", label) or "peo-rich" in label):
        loading = _number(row.get("pedot_pss_ratio_raw"))
        if loading is None:
            m = re.fullmatch(rf"\s*({_LC_NUMBER})\s*wt\.?\s*%\s*", _lc_text(row.get("pedot_pss_ratio_raw")))
            loading = float(m.group(1)) if m else None
        if loading is not None:
            out["cv_label_blend_reported_loading_scalar_unresolved_basis"] = loading
    if "centrifugal ultrafiltration" in process:
        out["cv_label_solvent_replacement_ultrafiltration"] = 1.
    if "redispersed" in process or "o-pedot" in label:
        out["cv_label_organic_redispersion"] = 1.
    if re.search(r"\bno\s+rinsing\b", label + "; " + post):
        out["cv_label_rinsing_condition"] = "none"
    elif re.search(r"\bcosolvent\s+rinsing\b", label + "; " + post):
        out["cv_label_rinsing_condition"] = "cosolvent"
    elif re.search(r"\bwater\s+rinsing\b", label + "; " + post):
        out["cv_label_rinsing_condition"] = "water"
    # Organic/water mixture ratios are retained in their stated volume basis.
    # A sum of 100 is required; no solvent-property or delivered film dose is
    # assigned. Distinct organic species and ratios are not collapsed together.
    mixture_values = defaultdict(list)
    for solvent, pattern in _LC_SOLVENTS.items():
        if solvent == "water":
            continue
        for text in (label, post):
            patterns = (
                rf"\b(?:{pattern})\s*/\s*(?:water|h2o)\s*\(?\s*({_LC_NUMBER})\s*/\s*({_LC_NUMBER})\s*vol\.?\s*%",
                rf"({_LC_NUMBER})\s*/\s*({_LC_NUMBER})\s*(?:{pattern})\s*/\s*(?:water|h2o)\b",
                rf"({_LC_NUMBER})\s*vol\.?\s*%\s*(?:{pattern})\s*/\s*({_LC_NUMBER})\s*vol\.?\s*%\s*(?:water|h2o)\b",
            )
            for pattern_pair in patterns:
                for match in re.finditer(pattern_pair, text):
                    organic, water = float(match.group(1)), float(match.group(2))
                    if 0 <= organic <= 100 and 0 <= water <= 100 and math.isclose(organic + water, 100., abs_tol=1e-7):
                        mixture_values["cosolvent_" + solvent + "_volpct"].append(organic)
                        mixture_values["cosolvent_water_volpct"].append(water)
                        out["cv_label_cosolvent_organic_component"] = solvent
    for name, values in mixture_values.items():
        _lc_store_unique(out, name, values)
    # Fixed optical/solvent input extras omitted by older allowlists.
    for item in row.get("extra_scalar_features") or []:
        if not isinstance(item, Mapping):
            continue
        name, unit, value = _slug(item.get("name")), _unit(item.get("unit")), _number(item.get("value"))
        if value is None or _FORBIDDEN.search(name):
            continue
        if name in {"hansen_dd", "hansen_dp", "hansen_dh"} and unit in {"mpa^0.5", "mpa0.5"}:
            out["cv_label_reported_" + name] = value
        elif name == "hz_concentration" and unit in {"vol%", "vol.%"}:
            out["cv_label_hydrazine_reported_volpct"] = value
    return out
