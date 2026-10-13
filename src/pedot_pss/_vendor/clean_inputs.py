"""Experiment-before-only input contract for a conductivity surrogate.

Projection is the first operation. Private dataset identifiers, targets, outcome
uncertainties and measured film characterization are never handed to encoders.
This module does no file lookup, training, or source-dependent feature recovery.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from functools import lru_cache

INPUT_VERSION = "experiment-input-v3-text-clean-1012"
NUMERIC_FIELDS = (
    "stock_solids_wt_percent", "dry_temperature_C", "dry_duration_min",
    "anneal_temperature_C", "anneal_duration_min", "post_treatment_temperature_C",
    "post_treatment_duration_min", "measurement_temperature_C",
)
TEXT_FIELDS = (
    "polymer_grade", "stock_solids_scope", "pedot_pss_ratio_raw", "dose_basis",
    "deposition_method", "substrate", "specimen_geometry", "post_treatment",
    "process_sequence", "measurement_method", "measurement_direction", "measurement_quantity",
)
INPUT_FIELDS = TEXT_FIELDS + NUMERIC_FIELDS + (
    "additive_names", "additive_amounts_raw", "extra_scalar_features", "process_stages",
)
# A closed allowlist. Measured film properties and electrical response proxies
# do not become inputs merely because a source calls them an "extra feature".
EXTRA_SCALAR_NAMES = frozenset({
    "coating_speed", "coating_stage_temperature", "post_coating_speed",
    "post_coating_stage_temperature", "post_dry_duration", "second_anneal_temperature",
    "second_anneal_duration", "spin_speed", "spin_duration", "pump_speed",
    "writing_speed", "motion_platform_speed", "laser_line_gap", "total_draw_ratio",
    "ph", "solution_ph", "concentration", "hz_concentration", "tdae_concentration",
    "tmu_concentration", "final_ink_solids_wt_percent", "working_solution_concentration",
    "organic_solvent_content", "pss_polymerization_time", "ultrasonication_time",
    "water_immersion_duration", "water_immersion_time", "treating_temperature",
    "vacuum_drying_duration", "post_wash_drying_temperature", "post_wash_drying_duration",
    "air_storage_time", "relative_humidity", "measurement_frequency", "applied_voltage",
    "ebl_acceleration_voltage", "nozzle_diameter", "nozzle_to_substrate_distance",
    "flow_rate", "drop_spacing", "mix_duration", "mix_rotation_speed", "blade_gap",
    "stock_volume", "substrate_uvo_duration", "post_dry_temperature",
    "planned_via_diameter", "measurement_bias", "blend_pedot_pss_dry_mass_percent",
    "preparation_water_calcium_mg_l", "preparation_water_sodium_mg_l",
    "preparation_water_magnesium_mg_l", "preparation_water_potassium_mg_l",
    "preparation_water_chloride_mg_l", "preparation_water_so4_reported_mg_l",
    "preparation_water_ph",
})
STAGE_NUMERIC_FIELDS = (
    "temperature_C", "duration_s", "speed_mm_s", "rotation_rpm", "blade_gap_um",
    "relative_humidity_percent", "pressure_bar", "flow_rate_mL_min", "voltage_V",
    "uv_wavelength_nm", "laser_power_W", "cycles",
)
STAGE_TEXT_FIELDS = ("kind", "method", "atmosphere", "pressure_reference", "object")
STAGE_KINDS = frozenset({"mix", "filter", "deposit", "dry", "anneal", "heat", "cool",
    "post_treat", "rinse", "wash", "immerse", "soak", "stretch", "press",
    "uv", "uv_ozone", "laser", "plasma", "store", "humidity", "other"})

_IDENTITY = re.compile(
    r"https?://|10\.\d{4,9}/\S+|\bdoi\b|(?:paper|source|row|sample|series|"
    r"specimen|experiment|batch)[_ -]?(?:id|no\.?|number)\b|"
    r"\b(?:N|B|O)\d{3}\b|\bXU_POLYBOT_2025\b|"
    r"(?:src|exp|ev|obs|state):[a-z0-9_-]+", re.I)
_RESULT = re.compile(
    r"\b(?:results?|targets?|predictions?|seebeck|resistivity|resistance|"
    r"coverage|crystallinity|coherence|raman|xps|roughness|mobility|carrier\s+density|"
    r"power[\s-]*factor|sheet\s*resist|standard\s*deviation|uncertainty|transmittance)\b|"
    r"[σΩ]|\bsigma\b|\bS\s*/\s*(?:cm|m)\b|\bS\s*(?:cm|m)\s*[-^]\s*1\b|"
    r"conductivit(?:y|ies)\s*(?:[:=]|was|is|of|reached|increased|decreased|reported|data|values|improved|unchanged)|"
    r"\b(?:did|does)\s+not\s+(?:alter|change|affect)\b.{0,120}\bconductivit(?:y|ies)\b|"
    r"\bgave\s+(?:the\s+)?same\s+enhancement\b|"
    r"\b(?:partly\s+dissolved|did\s+not\s+yield\s+a\s+good)\b", re.I)
_REFERENCE = re.compile(r"\b(?:fig(?:ure)?\.?|table|supplementary\s+(?:fig|table))\s*"
    r"[A-Za-z]?\d+(?:[A-Za-z]|\([^)]*\))?", re.I)
_PLANNED_MEASUREMENT = re.compile(
    r"(?:in\s*situ\s+)?conductivity\s+(?:measurement|characterization|monitor(?:ing|ed)?)", re.I)
_OUTCOME_CLAUSE = re.compile(r"\b(?:measured\s+(?:thickness|diameter)|thickness\s+(?:measured|by)|"
    r"final[- ]film\s+conductivity|optical\s+thickness|filtering\s+before\s+averaging|"
    r"electrical\s+positions|retained\s+n|author\s+train|train/test|"
    r"(?:initial|final)\s+(?:film\s+)?thickness|(?:thicker|thinner)\s+(?:initial|final)\s+film|average\s+pattern\s+height|"
    r"final\s+film\s+\d+(?:\.\d+)?\s*wt)\b", re.I)
_CITATION_CLAUSE = re.compile(
    r"\bref(?:erences?)?\.?\s*\d+|\b(?:assigned|supplied)\s+(?:files|SI)\b|"
    r"\b(?:this|the)\s+(?:paper|article)\s+(?:only|does|gives|states|reports|provides|shows)\b|"
    r"\b(?:article|paper)\s+(?:does|gives)\b|"
    r"\b(?:exact|detailed|individual|treatment|concentration)\b.{0,90}"
    r"\b(?:not\s+(?:reported|stated|specified|provided)|unavailable)\b|"
    r"\b(?:plotted\s+condition\s+distinguished|range\s+retained\s+in\s+evidence|"
    r"not\s+(?:(?:further|explicitly|numerically|separately)\s+)?(?:reported|stated|specified|provided|defined|assigned)|"
    r"no\s+rinse\s+reported|labeled\s+\d+(?:\.\d+)?|same\s+preparation\s+as$|companion\s+(?:XRR|AFM|GIWAXS)\b)\b", re.I)


def _strip_intake_annotations(text):
    """Remove extraction wording while preserving explicit recipe quantities.

    This is a pure input-language operation, without publication lookup or
    target access. Alternatives that were never assigned to the specimen are
    removed as whole clauses by the caller, rather than choosing one option.
    """
    text = _REFERENCE.sub("", text)
    text = re.sub(r"\b(?:Scheme)\s*\d+[A-Za-z]?\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:this|the present)\s+(?:paper|article|study)\s+(?:only\s+)?"
                  r"(?:gives|provides|reports)\s+(?:the\s+)?(?:qualitative\s+)?"
                  r"(?:summary|description)\s+of\s+", "", text, flags=re.I)
    text = re.sub(r"\b(?:paper|article)\s+(?:states|reports)\s+", "", text, flags=re.I)
    text = re.sub(r"\bpaper\s+gives\s+a\s+general\s+", "", text, flags=re.I)
    text = re.sub(r"\s+rather\s+than\s+one\s+sample-specific\s+value\b", "", text, flags=re.I)
    text = re.sub(r"\breports\s+stock\b", "stock", text, flags=re.I)
    text = re.sub(r"\b(?:article\s+text\s*:|labels\s+the\s+formulation)\s*", "", text, flags=re.I)
    text = re.sub(r"\s*,?\s*as\s+(?:stated|reported)\s+(?:by|in)\s+(?:the\s+)?(?:paper|article)\b", "", text, flags=re.I)
    text = re.sub(r"\bstated\s+in\s+main\s+text\s+as\s+", "", text, flags=re.I)
    text = re.sub(r"\b(?:in\s+SI|per\s+stated\s+film\s+protocol)\b", "", text, flags=re.I)
    text = re.sub(r"\([^()]*\b(?:digitized|x-axis|Supplementary|main\s+text|SI\s+method)\b[^()]*\)", "", text, flags=re.I)
    text = re.sub(r"\s*,?\s*(?:as\s+(?:read\s+from|labeled(?:\s+(?:on\s+the\s+x-axis|in\s+SI|in))?|plotted)|"
                  r"(?:x-axis\s+)?digitized\s+from(?:\s+x-axis)?|shown\s+in|plotted\s+in\s+Supplementary)"
                  r"\b", "", text, flags=re.I)
    text = re.sub(r"\s+used\s+as\s+plotted\s+condition\b", "", text, flags=re.I)
    text = re.sub(r"\s+as\s+shown\b[^;]*", "", text, flags=re.I)
    text = re.sub(r"\bx-axis\s+gives\s+", "", text, flags=re.I)
    text = re.sub(r"\s*,?\s*as\s+labeled\b", "", text, flags=re.I)
    text = re.sub(r"\blabels\s+the\s+double\s+layer\s+as\s+([^;]*?),\s*\d+(?:\.\d+)?\s*nm\b", r"\1", text, flags=re.I)
    text = re.sub(r"\bHighest[- ]power[- ]factor\s+", "", text, flags=re.I)
    text = re.sub(r"\s+at\s+optimum\s+power\b[^;]*", "", text, flags=re.I)
    text = re.sub(r"\bspin-coated\s+thicker\s+film\b", "spin-coated film", text, flags=re.I)
    # Companion characterization specimens and qualitative film dimensions
    # are not settings of the conductivity experiment. Substrate dimensions
    # (e.g. foil thickness) remain usable design information.
    text = re.sub(r"\bcleaned\s+silicon\s*\(GIWAXS\)\s*;\s*", "", text, flags=re.I)
    text = re.sub(r"\s*;\s*silicon\s+used\s+for\s+thick\s+XRD\s+specimens", "", text, flags=re.I)
    text = re.sub(r"\s*\(intrinsic\s+Si\s+for\s+conductivity/GIXRD/AFM\)?", "", text, flags=re.I)
    text = re.sub(r"\brepeat\s+thermoelectric\s+characterization\s+and\s+perform\s+UV-vis/GIWAXS\b", "", text, flags=re.I)
    text = re.sub(r"\bfilm\s+thickness\s+varied\b", "", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:nm|[uμµ]m)[ -]thick\s+film\b", "film", text, flags=re.I)
    text = re.sub(r"\b(?:thicker|thick)\s+(?=(?:PEDOT/PSS\s+)?films?\b)", "", text, flags=re.I)
    text = re.sub(r"\binhomogeneous\s+self-supported\b", "self-supported", text, flags=re.I)
    text = re.sub(r"\bPSSA-[12]\b", "PSSA", text)
    text = re.sub(r"\b(?:same\s+(?:procedure|synthesis)\s+as\s+)?(?:Pre\s+)?T-ECH(?:\s+\d+)?\b", "", text, flags=re.I)
    text = re.sub(r"\bPH\s*1000-\d+(?:\.\d+)?%P(?:-\d+(?:\.\d+)?%EG)?\b", "PH1000", text, flags=re.I)
    text = re.sub(r"\((?:\s*)\)", "", text)
    return text.strip(" ,;.")


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not re.fullmatch(
            r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*", value):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _text(value):
    if not isinstance(value, str):
        return None
    s = unicodedata.normalize("NFKC", value).replace("−", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else None


def sanitize_text(value, *, process=False):
    """Keep bounded condition clauses, discard identifier/outcome clauses.

    This is not a parser for sample labels. Only explicitly allowlisted recipe
    fields call it. Literal quantities in retained condition clauses survive;
    measured responses, reference numbers and source IDs do not.
    """
    text = _text(value)
    if text is None:
        return None
    return _sanitize_normalized_text(text, process)


@lru_cache(maxsize=32768)
def _sanitize_normalized_text(text, process):
    # Repeated recipe fields use the same pure projection. This process-local
    # memo retains no files or model weights and never enters fitted state.
    text = _strip_intake_annotations(text)
    # These are article-dependent labels, not independently specified
    # chemistry. Ingredient/quantity fields and explicit operations survive.
    if re.fullmatch(r"Ox[1-5]", text, re.I):
        return "unspecified additive"
    if process:
        text = re.sub(r"\b(?:Pe/AP|BAP|AP)\b", "", text)
        text = re.sub(r"(?<!on )(?<!onto )(?<!flexible )\bPP(?=\s*(?:;|$|film|fibers))", "", text)
        text = re.sub(r"\b(?:LP|SL)(?=\s+(?:film|hydrogel))", "", text)
        text = re.sub(r"\bP_DMSO\b", "", text)
        if not re.search(r"Fine\s+Turner", text, re.I) and re.search(r"(?:polymerization|particles\s+subjected\s+to).{0,80}\bFT[0-3]\b", text, re.I):
            text = re.sub(r"\bFT[0-3]\b", "", text)
        text = re.sub(r"\bfilm\s+film\b", "film", text)
        text = re.sub(r",\s*,", ",", text)
    accepted = []
    for clause in re.split(r";|→|->|\n|(?<=[.!?])\s+(?=[A-Z])", text):
        clause = clause.strip(" ,;.")
        if not clause or _IDENTITY.search(clause) or _RESULT.search(clause) or _OUTCOME_CLAUSE.search(clause) or _CITATION_CLAUSE.search(clause):
            continue
        clause = _PLANNED_MEASUREMENT.sub("electrical measurement", clause)
        # Categorical solvent-number codes are an author indexing convention;
        # solvent names and compositions in separate clauses are preserved.
        clause = re.sub(r"\bPost[- ]processing solvent No\.\s*\d+\s*\(categorical\)", "", clause, flags=re.I)
        clause = re.sub(r"\s+", " ", clause).strip(" ,;.")
        clause = re.sub(r"\s+(?:to\s+obtain|to\s+make|to)$", "", clause, flags=re.I)
        if clause:
            accepted.append(clause[:1000])
    result = "; ".join(accepted)
    return result[:4000] or None


def _geometry(value):
    """Keep the intended specimen type, never an observed film dimension."""
    text = sanitize_text(value) or ""
    patterns = (("vertical junction", r"vertical|via|out.of.plane"),
                ("film", r"film|coating"), ("fiber", r"fiber|fibre|filament|yarn"),
                ("printed trace", r"trace|printed|line"), ("bulk", r"bulk|pellet|block"),
                ("gel", r"gel"), ("porous", r"porous|foam|aerogel"))
    found = [name for name, pattern in patterns if re.search(pattern, text, re.I)]
    return "; ".join(found) or None


def _measurement(value):
    # Retain the planned electrical method when intake joined it with an
    # optical-thickness characterization narrative using "and".
    raw = _text(value) or ""
    raw = re.split(r"\band\s+optical\s+thickness\b", raw, maxsplit=1, flags=re.I)[0]
    text = sanitize_text(raw) or ""
    patterns = (("four-point probe", r"four.point|4.point|4pp"),
                ("two-point probe", r"two.point|2.point|two.terminal"),
                ("van der Pauw", r"van.der.pauw"), ("Hall", r"\bhall\b"),
                ("impedance", r"impedance|\beis\b"), ("DC", r"\bdc\b|direct.current"),
                ("in-plane", r"in.plane|lateral"), ("out-of-plane", r"out.of.plane|through.plane|vertical"),
                ("ionic", r"\bionic\b|protonic"), ("electronic", r"\belectronic\b"),
                ("bias sweep", r"bias\s+sweep|voltage\s+sweep"))
    parts = [name for name, pattern in patterns if re.search(pattern, text, re.I)]
    # Explicit planned bias/frequency survive; arbitrary instrument narrative
    # and observed spatial SD/counts are not used as a source fingerprint.
    for pattern in (r"(?:at|maximum|bias)\s*[+-]?\d+(?:\.\d+)?\s*V\b",
                    r"\d+(?:\.\d+)?\s*(?:kHz|MHz|Hz)\b"):
        parts.extend(match.group(0) for match in re.finditer(pattern, text, re.I))
    return "; ".join(parts) or None


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _extras(value):
    rows = []
    if not isinstance(value, (list, tuple)):
        return rows
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _slug(item.get("name"))
        number = _number(item.get("value"))
        unit = sanitize_text(item.get("unit"))
        if name in EXTRA_SCALAR_NAMES and number is not None:
            rows.append({"name": name, "value": number, "unit": unit})
    # Locator, original row order and notes cannot be indirect features.
    return sorted(rows, key=lambda x: (x["name"], x["unit"] or "", x["value"]))


def _stages(value):
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        kind = _slug(item.get("kind"))
        if kind not in STAGE_KINDS:
            continue
        stage = {key: sanitize_text(item.get(key), process=True) for key in STAGE_TEXT_FIELDS}
        stage["kind"] = kind
        for key in STAGE_NUMERIC_FIELDS:
            stage[key] = _number(item.get(key))
        for key in ("temperature_range_C", "duration_range_s"):
            bounds = item.get(key)
            stage[key] = ([float(n) for n in bounds] if isinstance(bounds, (list, tuple))
                and len(bounds) == 2 and all(_number(n) is not None for n in bounds)
                and float(bounds[0]) <= float(bounds[1]) else None)
        out.append(stage)
    return out


def _quantity(quantity, target_unit):
    if not isinstance(quantity, Mapping) or quantity.get("status") != "reported":
        return None
    value = quantity.get("value")
    if not isinstance(value, Mapping) or value.get("kind") != "point":
        return None
    raw = _number(value.get("value"))
    unit = str(quantity.get("unit", "")).casefold().replace("μ", "u").replace("µ", "u")
    factors = {
        "C": {"degc": 1, "c": 1, "°c": 1},
        "s": {"s": 1, "sec": 1, "min": 60, "h": 3600},
        "mm_s": {"mm/s": 1, "mm/sec": 1, "m/s": 1000, "mm/min": 1/60},
        "um": {"um": 1, "nm": .001, "mm": 1000},
        "rpm": {"rpm": 1}, "mL": {"ml": 1, "ul": .001},
    }
    if raw is not None and target_unit == "C" and unit in {"k", "kelvin"}:
        return raw - 273.15
    factor = factors.get(target_unit, {}).get(unit)
    return raw * factor if raw is not None and factor is not None else None


def _native_text(value):
    return value.get("value") if isinstance(value, Mapping) and value.get("status") == "reported" else None


def _native_amount(quantity):
    if not isinstance(quantity, Mapping) or quantity.get("status") != "reported":
        return None
    point = quantity.get("value", {})
    number = _number(point.get("value")) if isinstance(point, Mapping) and point.get("kind") == "point" else None
    unit = sanitize_text(quantity.get("unit"))
    return f"{number:g} {unit}" if number is not None and unit else None


def _native_projection(record):
    """Resolve local native material references, discard all identifier strings."""
    experiment = record.get("experiment") if isinstance(record, Mapping) else None
    if not isinstance(experiment, Mapping):
        return {}
    materials = {m.get("material_id"): m for m in experiment.get("materials", []) if isinstance(m, Mapping)}
    result, names, amounts, post = {}, [], [], []
    for form in experiment.get("formulations", []):
        for item in form.get("constituents", []):
            material = materials.get(item.get("material_id"), {})
            name = sanitize_text(_native_text(material.get("name")))
            amount = _native_amount(item.get("amount"))
            if item.get("role") == "polymer":
                grade = sanitize_text(_native_text(material.get("grade")))
                supplier = sanitize_text(_native_text(material.get("supplier")))
                if grade:
                    result["polymer_grade"] = f"{grade} ({supplier})" if supplier else grade
                volume = _quantity(item.get("amount"), "mL")
                if volume is not None:
                    result.setdefault("extra_scalar_features", []).append({"name": "stock_volume", "value": volume, "unit": "mL"})
            elif item.get("role") == "additive" and name:
                names.append(name); amounts.append(amount)
                basis = item.get("basis", {}).get("denominator", {})
                if isinstance(basis, Mapping) and basis.get("status") == "reported":
                    result["dose_basis"] = sanitize_text(basis.get("value"))
                else:
                    result.setdefault("dose_basis", "vol%; stock_or_mixture_denominator_unresolved")
            elif item.get("role") == "post_treatment_agent" and name:
                post.append(f"{name} {amount}" if amount else name)
    if names:
        result["additive_names"], result["additive_amounts_raw"] = names, amounts
    if post:
        result["post_treatment"] = "blade-applied solvent post-treatment; " + "; ".join(post)
    substrate = sanitize_text(_native_text(experiment.get("substrate")))
    if substrate:
        result["substrate"] = substrate
    prep = sanitize_text(_native_text(experiment.get("substrate_preparation")))
    stages = []
    if prep:
        match = re.search(r"UV.?ozone.*?(\d+(?:\.\d+)?)\s*min", prep, re.I)
        stages.append({"kind": "uv_ozone", "method": prep, "duration_s": float(match.group(1))*60 if match else None})
    counts = {"dry": 0, "anneal": 0}
    parameter_map = {"temperature": ("temperature_C", "C"), "duration": ("duration_s", "s"),
                     "blade_speed": ("speed_mm_s", "mm_s"), "rotation_speed": ("rotation_rpm", "rpm"),
                     "blade_gap": ("blade_gap_um", "um")}
    for step in sorted(experiment.get("steps", []), key=lambda s: _number(s.get("order")) or 0):
        kind = _slug(step.get("kind"))
        if kind not in STAGE_KINDS:
            continue
        stage = {"kind": kind, "method": sanitize_text(_native_text(step.get("method")), process=True)}
        for parameter in step.get("parameters", []):
            mapping = parameter_map.get(parameter.get("name"))
            if mapping:
                key, unit = mapping
                stage[key] = _quantity(parameter.get("quantity"), unit)
        stages.append(stage)
        if kind == "deposit":
            mode = sanitize_text(step.get("deposition_mode"))
            result["deposition_method"] = f"{mode} coating" if mode else stage["method"]
            for key, name, unit in (("speed_mm_s", "coating_speed", "mm/s"), ("temperature_C", "coating_stage_temperature", "C"), ("blade_gap_um", "blade_gap", "um")):
                if stage.get(key) is not None:
                    result.setdefault("extra_scalar_features", []).append({"name": name, "value": stage[key], "unit": unit})
        if kind == "post_treat":
            result["post_treatment_temperature_C"] = stage.get("temperature_C")
            result["post_treatment_duration_min"] = stage["duration_s"]/60 if stage.get("duration_s") is not None else None
            for key, name, unit in (("speed_mm_s", "post_coating_speed", "mm/s"), ("temperature_C", "post_coating_stage_temperature", "C")):
                if stage.get(key) is not None:
                    result.setdefault("extra_scalar_features", []).append({"name": name, "value": stage[key], "unit": unit})
        if kind in counts:
            counts[kind] += 1
            if counts[kind] == 1:
                result[f"{kind}_temperature_C"] = stage.get("temperature_C")
                result[f"{kind}_duration_min"] = stage["duration_s"]/60 if stage.get("duration_s") is not None else None
            elif kind == "dry":
                for key, name, unit in (("duration_s", "post_dry_duration", "s"), ("temperature_C", "post_dry_temperature", "C")):
                    if stage.get(key) is not None:
                        result.setdefault("extra_scalar_features", []).append({"name": name, "value": stage[key], "unit": unit})
            else:
                for key, name, unit in (("temperature_C", "second_anneal_temperature", "C"), ("duration_s", "second_anneal_duration", "s")):
                    if stage.get(key) is not None:
                        result.setdefault("extra_scalar_features", []).append({"name": name, "value": stage[key], "unit": unit})
    result["process_stages"] = stages
    result["specimen_geometry"] = "film"
    # Neither native states nor observations are inspected.
    return result


def _stage_controls(clean):
    """Make structured stages usable by the same legacy recipe coordinates."""
    extras = {(item["name"], item["unit"]): item for item in clean["extra_scalar_features"]}
    counts = {"dry": 0, "anneal": 0}
    for stage in clean["process_stages"]:
        kind = stage["kind"]
        def extra(name, field, unit):
            if stage.get(field) is not None:
                extras[(name, unit)] = {"name": name, "value": stage[field], "unit": unit}
        if kind in counts:
            counts[kind] += 1
            if counts[kind] == 1:
                if clean[f"{kind}_temperature_C"] is None:
                    clean[f"{kind}_temperature_C"] = stage.get("temperature_C")
                if clean[f"{kind}_duration_min"] is None and stage.get("duration_s") is not None:
                    clean[f"{kind}_duration_min"] = stage["duration_s"] / 60
            elif kind == "dry":
                extra("post_dry_duration", "duration_s", "s")
                extra("post_dry_temperature", "temperature_C", "C")
            else:
                extra("second_anneal_temperature", "temperature_C", "C")
                extra("second_anneal_duration", "duration_s", "s")
        elif kind == "deposit":
            extra("coating_speed", "speed_mm_s", "mm/s")
            extra("coating_stage_temperature", "temperature_C", "C")
            extra("blade_gap", "blade_gap_um", "um")
        elif kind == "post_treat":
            if clean["post_treatment_temperature_C"] is None:
                clean["post_treatment_temperature_C"] = stage.get("temperature_C")
            if clean["post_treatment_duration_min"] is None and stage.get("duration_s") is not None:
                clean["post_treatment_duration_min"] = stage["duration_s"] / 60
            extra("post_coating_speed", "speed_mm_s", "mm/s")
            extra("post_coating_stage_temperature", "temperature_C", "C")
        elif kind == "mix":
            extra("mix_duration", "duration_s", "s")
            extra("mix_rotation_speed", "rotation_rpm", "rpm")
        elif kind == "uv_ozone":
            extra("substrate_uvo_duration", "duration_s", "s")
    clean["extra_scalar_features"] = sorted(extras.values(), key=lambda x: (x["name"], x["unit"] or "", x["value"]))
    return clean


def project_experiment(original_row, native_record=None):
    """Return exactly 24 allowlisted fields with nulls and empty lists explicit.

    Dataset envelopes may be supplied; only original_row is projected. A native
    experiment may be passed explicitly for training ingestion, without a file
    lookup. Source and row identifiers used for joining remain outside this API.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("experiment input must be a mapping")
    raw = original_row.get("original_row", original_row)
    if not isinstance(raw, Mapping):
        raise TypeError("original_row must be a mapping")
    out = {key: sanitize_text(raw.get(key), process=key in {"deposition_method", "post_treatment", "process_sequence"}) for key in TEXT_FIELDS}
    out.update({key: _number(raw.get(key)) for key in NUMERIC_FIELDS})
    out["specimen_geometry"] = _geometry(raw.get("specimen_geometry"))
    out["measurement_method"] = _measurement(raw.get("measurement_method"))
    names = raw.get("additive_names") or []
    amounts = raw.get("additive_amounts_raw") or []
    if isinstance(names, str): names = [names]
    if isinstance(amounts, (str, int, float)): amounts = [amounts]
    names = names if isinstance(names, (list, tuple)) else []
    amounts = amounts if isinstance(amounts, (list, tuple)) else []
    aligned = len(names) == len(amounts)
    pairs = [(sanitize_text(name), sanitize_text(str(amounts[i])) if aligned and amounts[i] is not None else None)
             for i, name in enumerate(names)]
    pairs = [(name, amount) for name, amount in pairs if name]
    out["additive_names"] = [name for name, _ in pairs]
    out["additive_amounts_raw"] = [amount for _, amount in pairs]
    out["extra_scalar_features"] = _extras(raw.get("extra_scalar_features"))
    out["process_stages"] = _stages(raw.get("process_stages"))
    native = native_record if native_record is not None else (original_row if isinstance(original_row.get("experiment"), Mapping) else None)
    if native is not None:
        projected = _native_projection(native)
        native_extras = projected.pop("extra_scalar_features", [])
        for key, value in projected.items():
            out[key] = value
        extras = {(e["name"], e["unit"]): e for e in out["extra_scalar_features"]}
        extras.update({(e["name"], e["unit"]): e for e in _extras(native_extras)})
        out["extra_scalar_features"] = sorted(extras.values(), key=lambda x: (x["name"], x["unit"] or "", x["value"]))
        out["process_stages"] = _stages(out["process_stages"])
        out["process_sequence"] = _stage_text(out["process_stages"])
    _stage_controls(out)
    return {key: out[key] for key in INPUT_FIELDS}


canonicalize = project_experiment


def _stage_text(stages):
    clauses = []
    for stage in stages:
        values = [stage["kind"]]
        if stage.get("method"):
            values.append(stage["method"])
        units = {"temperature_C": "C", "duration_s": "s", "speed_mm_s": "mm/s",
                 "rotation_rpm": "rpm", "blade_gap_um": "um", "relative_humidity_percent": "% RH",
                 "pressure_bar": "bar", "flow_rate_mL_min": "mL/min", "voltage_V": "V",
                 "uv_wavelength_nm": "nm", "laser_power_W": "W", "cycles": "cycles"}
        for key, unit in units.items():
            if stage.get(key) is not None:
                values.append(f"{key}={stage[key]:g} {unit}")
        for key in ("temperature_range_C", "duration_range_s"):
            if stage.get(key):
                lo, hi = stage[key]
                values.append(f"{key}={lo:g} to {hi:g}")
        for key in ("atmosphere", "pressure_reference", "object"):
            if stage.get(key): values.append(f"{key}={stage[key]}")
        clauses.append(" ".join(values))
    return "; ".join(clauses) or None


def original_view(canonical):
    """Safe legacy feature-builder row, with ordered stages serialized once."""
    clean = project_experiment(canonical)
    # Original MatBERT pipeline's explicit input-language synonym. Preserve
    # the physical meaning and prevent an all-[UNK] field in the cased model.
    if clean["dose_basis"] == "成膜前未加添加剂":
        clean["dose_basis"] = "no additive added before film formation"
    if clean["process_stages"]:
        stage_text = _stage_text(clean["process_stages"])
        # Native/user stage lists are authoritative; do not silently duplicate
        # their thermal steps with an additional prose history.
        clean["process_sequence"] = stage_text
    return clean


def text_view(canonical):
    """Deterministic text representing only raw experimental conditions."""
    clean = original_view(canonical)
    return "\n".join(f"{key}: {json.dumps(clean[key], ensure_ascii=False, sort_keys=True)}"
                     for key in INPUT_FIELDS if clean[key] is not None and clean[key] != [])


def feature_view(canonical):
    """Reuse scientific recipe builders only after irreversible projection.

    Callers bundle these existing modules beside this one. Their imports contain
    no dataset lookup; the chemical descriptor table is a fixed model resource.
    """
    from .features import build_features
    from .identity_features import enrich_identity_features
    from .process_features import enrich_process_features
    from .quantity_features import quantity_features
    from .polybot_features import polybot_features
    clean = original_view(canonical)
    out = build_features(clean, include_observed=False, include_context=False)
    out = enrich_identity_features(clean, out)
    out = enrich_process_features(clean, out)
    out.update(quantity_features(clean))
    out.update(polybot_features(clean))
    # The boundary owns its closed allowlist, including newly recovered design
    # and starting-water controls. An older feature builder need not know a
    # new field for it to become a real, unit-labelled predictor covariate.
    declared = {}
    for item in clean["extra_scalar_features"]:
        unit = "pct" if item["unit"] == "%" else (_slug(item["unit"] or "unreported") or "unreported")
        key = "extra_" + item["name"] + "__" + unit
        declared.setdefault(key, set()).add(item["value"])
    for key, values in declared.items():
        if len(values) == 1:
            out[key] = next(iter(values))
            out["has_" + key] = 1.0
        else:
            out.pop(key, None)
            out["has_" + key] = 0.0
            out["conflicting_" + key] = 1.0
    # These are explicitly supplied controls, not guessed values from sample
    # labels. The old Polybot regex does not recognize every structured stage.
    aliases = {"stock_volume": "stock_volume_mL", "mix_duration": "mix_duration_s",
        "mix_rotation_speed": "mix_rotation_rpm", "blade_gap": "blade_gap_um",
        "substrate_uvo_duration": "substrate_uvo_duration_s",
        "post_dry_temperature": "post_dry_stage_temperature_C"}
    for item in clean["extra_scalar_features"]:
        if item["name"] in aliases:
            key = aliases[item["name"]]
            expected_unit = {"stock_volume": "mL", "mix_duration": "s", "mix_rotation_speed": "rpm",
                "blade_gap": "um", "substrate_uvo_duration": "s", "post_dry_temperature": "C"}[item["name"]]
            if item["unit"] == expected_unit:
                out["pb_" + key] = item["value"]
                out["pb_has_" + key] = 1.0
    # Explicit stages retain every known experimental scalar, including native
    # shared protocol constants which older free-text regexes might not parse.
    for index, stage in enumerate(clean["process_stages"]):
        prefix = f"planned_stage_{index:02d}_"
        for key in STAGE_TEXT_FIELDS:
            if stage.get(key) is not None:
                out[prefix + key] = stage[key]
        for key in STAGE_NUMERIC_FIELDS:
            out["has_" + prefix + key] = float(stage.get(key) is not None)
            if stage.get(key) is not None:
                out[prefix + key] = stage[key]
    return out


build_scientific_features = feature_view
physics_view = feature_view
