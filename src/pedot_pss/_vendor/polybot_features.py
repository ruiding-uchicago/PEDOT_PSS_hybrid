"""Compact Polybot recipe inputs with stage-aware native JSON support.

Only seven author INPUT columns, native formulations/process steps, and the
corresponding merged recipe fields are used. Outcomes, coverage, uncertainties,
historical Train/Test labels and specimen identifiers are never features.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping

POLYBOT_FEATURE_VERSION = "polybot-input-v1"
CONTROL_COLUMNS = {
    "dmso_volpct_reported": "DMSO concentration (Vol%)",
    "eg_volpct_reported": "EG concentration (Vol%)",
    "first_coating_speed_mm_s": "Coating speed (mm/sec)",
    "first_coating_stage_temperature_C": "Coating temperature (Celsius)",
    "post_speed_mm_s": "Post coating speed (mm/sec)",
    "post_stage_temperature_C": "Post coating temperature (Celsius)",
}
SOLVENT_COLUMN = "Post-processing solvent No."
POLYBOT_KERNEL_FEATURES = tuple("pb_" + name for name in CONTROL_COLUMNS) + (
    "pb_post_methanol_volpct", "pb_post_ethanol_volpct", "pb_post_water_volpct")
# Verified against every native formulation in the 75-record intake. These are
# categories, not an ordinal solvent axis. Tuple order is methanol/ethanol/water.
POST_SOLVENT_RECIPES = {
    "1": (100., 0., 0.), "2": (20., 80., 0.), "3": (40., 60., 0.),
    "4": (50., 50., 0.), "5": (60., 40., 0.), "6": (80., 20., 0.),
    "7": (0., 100., 0.), "8": (80., 0., 20.),
}
CORE_KEYS = tuple(CONTROL_COLUMNS) + (
    "post_methanol_volpct", "post_ethanol_volpct", "post_water_volpct",
    "stock_volume_mL", "mix_duration_s", "mix_rotation_rpm", "blade_gap_um",
    "substrate_uvo_duration_s", "first_dry_stage_temperature_C", "first_dry_duration_s",
    "first_anneal_temperature_C", "first_anneal_duration_s",
    "post_dry_stage_temperature_C", "post_dry_duration_s",
    "second_anneal_temperature_C", "second_anneal_duration_s",
    "first_coating_duration_s", "post_application_duration_s",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and not re.fullmatch(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*", value):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _quantity(quantity: Any, unit: str) -> float | None:
    if not isinstance(quantity, Mapping) or quantity.get("status") != "reported":
        return None
    value = quantity.get("value")
    if not isinstance(value, Mapping) or value.get("kind") != "point":
        return None
    raw_unit = str(quantity.get("unit", "")).lower().replace("μ", "u").replace("µ", "u")
    factors = {
        "s": {"s": 1., "min": 60.}, "C": {"degc": 1., "c": 1., "°c": 1.},
        "mm_s": {"mm/s": 1., "mm/sec": 1., "m/s": 1000.},
        "um": {"um": 1., "nm": .001, "mm": 1000.},
        "rpm": {"rpm": 1.}, "mL": {"ml": 1., "ul": .001},
        "volpct": {"vol%": 1., "vol.%": 1.},
    }
    number = _number(value.get("value"))
    factor = factors.get(unit, {}).get(raw_unit)
    return number * factor if number is not None and factor is not None else None


def _put(out: dict, name: str, value: Any) -> None:
    number = _number(value)
    out["pb_has_" + name] = float(number is not None)
    if number is not None:
        out["pb_" + name] = number


def _native_inputs(row: Mapping) -> tuple[dict, str | None]:
    native = row.get("experiment")
    if not isinstance(native, Mapping):
        return {}, None
    inputs = {}
    raw = row.get("raw_csv_record", {}).get("original_row", {})
    # Whitelist only the seven author recipe controls; NEVER iterate all raw
    # columns, because the same native object also contains outcomes and IDs.
    for name, column in CONTROL_COLUMNS.items():
        if isinstance(raw, Mapping):
            inputs[name] = _number(raw.get(column))
    code = str(raw.get(SOLVENT_COLUMN, "")).strip() if isinstance(raw, Mapping) else ""
    materials = {material.get("material_id"): str(material.get("name", {}).get("value", "")).lower()
                 for material in native.get("materials", []) if isinstance(material, Mapping)}
    for formulation in native.get("formulations", []):
        for item in formulation.get("constituents", []):
            name = materials.get(item.get("material_id"), "")
            role = item.get("role")
            if role == "post_treatment_agent" and name in {"methanol", "ethanol", "water"}:
                inputs[f"post_{name}_volpct"] = _quantity(item.get("amount"), "volpct")
            if role == "polymer":
                inputs["stock_volume_mL"] = _quantity(item.get("amount"), "mL")
            if role == "additive":
                key = "dmso" if name == "dmso" else "eg" if name == "ethylene glycol" else None
                if key and inputs.get(f"{key}_volpct_reported") is None:
                    inputs[f"{key}_volpct_reported"] = _quantity(item.get("amount"), "volpct")
    occurrence = {"dry": 0, "anneal": 0}
    for step in sorted(native.get("steps", []), key=lambda item: item.get("order", 0)):
        kind = step.get("kind")
        if kind in occurrence:
            occurrence[kind] += 1
        prefix = ("mix" if kind == "mix" else "first_coating" if kind == "deposit" else
                  "post" if kind == "post_treat" else
                  ("first_dry" if occurrence["dry"] == 1 else "post_dry") if kind == "dry" else
                  ("first_anneal" if occurrence["anneal"] == 1 else "second_anneal") if kind == "anneal" else None)
        if prefix is None:
            continue
        for parameter in step.get("parameters", []):
            name = parameter.get("name")
            quantity = parameter.get("quantity")
            if name == "duration":
                key = "post_application_duration_s" if kind == "post_treat" else f"{prefix}_duration_s"
                inputs[key] = _quantity(quantity, "s")
            elif name == "temperature":
                key = (f"{prefix}_stage_temperature_C" if kind in {"deposit", "post_treat", "dry"}
                       else f"{prefix}_temperature_C")
                value = _quantity(quantity, "C")
                if value is not None:
                    inputs[key] = value
            elif name == "blade_speed":
                inputs[f"{prefix}_speed_mm_s"] = _quantity(quantity, "mm_s")
            elif name == "blade_gap":
                inputs["blade_gap_um"] = _quantity(quantity, "um")
            elif name == "rotation_speed" and kind == "mix":
                inputs["mix_rotation_rpm"] = _quantity(quantity, "rpm")
    substrate = native.get("substrate_preparation", {}).get("value", "")
    match = re.search(r"UV.?ozone.*?(\d+(?:\.\d+)?)\s*min", substrate, re.I)
    if match:
        inputs["substrate_uvo_duration_s"] = float(match.group(1)) * 60
    return inputs, code or None


def _merged_inputs(row: Mapping) -> tuple[dict, str | None]:
    inputs = {}
    names = row.get("additive_names") or []
    amounts = row.get("additive_amounts_raw") or []
    if isinstance(names, list) and isinstance(amounts, list) and len(names) == len(amounts):
        for name, amount in zip(names, amounts):
            normalized = str(name).lower().strip()
            chemical = "dmso" if normalized in {"dmso", "dimethyl sulfoxide", "dimethyl sulfoxide (dmso)"} else (
                "eg" if normalized in {"eg", "ethylene glycol", "ethylene glycol (eg)"} else None)
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*vol\.?\s*%\s*", str(amount), re.I)
            if chemical and match:
                inputs[f"{chemical}_volpct_reported"] = float(match.group(1))
    extras = {str(item.get("name")): item for item in row.get("extra_scalar_features") or []
              if isinstance(item, Mapping)}
    for field, name, accepted_units, factor in (
        ("coating_speed", "first_coating_speed_mm_s", {"mm/s", "mm/sec"}, 1.),
        ("post_coating_speed", "post_speed_mm_s", {"mm/s", "mm/sec"}, 1.),
        ("coating_stage_temperature", "first_coating_stage_temperature_C", {"c", "degc", "°c"}, 1.),
        ("post_coating_stage_temperature", "post_stage_temperature_C", {"c", "degc", "°c"}, 1.),
        ("post_dry_duration", "post_dry_duration_s", {"s"}, 1.),
        ("second_anneal_temperature", "second_anneal_temperature_C", {"c", "degc", "°c"}, 1.),
        ("second_anneal_duration", "second_anneal_duration_s", {"s"}, 1.),
    ):
        item = extras.get(field, {})
        if str(item.get("unit", "")).lower() in accepted_units and _number(item.get("value")) is not None:
            inputs[name] = _number(item["value"]) * factor
    for field, name, factor in (
        ("dry_temperature_C", "first_dry_stage_temperature_C", 1.),
        ("dry_duration_min", "first_dry_duration_s", 60.),
        ("anneal_temperature_C", "first_anneal_temperature_C", 1.),
        ("anneal_duration_min", "first_anneal_duration_s", 60.),
        ("post_treatment_temperature_C", "post_stage_temperature_C", 1.),
    ):
        number = _number(row.get(field))
        if number is not None:
            inputs.setdefault(name, number * factor)
    text = str(row.get("post_treatment") or "")
    for solvent in ("methanol", "ethanol", "water"):
        match = re.search(rf"\b{solvent}\s+(\d+(?:\.\d+)?)\s*vol\.?\s*%", text, re.I)
        if match:
            inputs[f"post_{solvent}_volpct"] = float(match.group(1))
    code_match = re.search(r"solvent\s+No\.\s*(\d+)\s*\(categorical\)", text, re.I)
    sequence = str(row.get("process_sequence") or "")
    for pattern, key, factor in (
        (r"UV.?ozone[^;]*?\((\d+(?:\.\d+)?)\s*min\)", "substrate_uvo_duration_s", 60.),
        (r"with\s+(\d+(?:\.\d+)?)\s*mL\s+pristine", "stock_volume_mL", 1.),
        (r"\((\d+(?:\.\d+)?)\s*rpm,", "mix_rotation_rpm", 1.),
        (r"rpm,\s*(\d+(?:\.\d+)?)\s*s\)", "mix_duration_s", 1.),
        (r"\((\d+(?:\.\d+)?)\s*(?:um|μm|µm)\s+gap", "blade_gap_um", 1.),
    ):
        match = re.search(pattern, sequence, re.I)
        if match:
            inputs[key] = float(match.group(1)) * factor
    if "continued post-coating stage setpoint" in sequence.lower() and "post_stage_temperature_C" in inputs:
        inputs["post_dry_stage_temperature_C"] = inputs["post_stage_temperature_C"]
    return inputs, code_match.group(1) if code_match else None


def polybot_features(original_row: Mapping[str, Any]) -> dict[str, float | str]:
    """Return compact recipe inputs for native or merged Polybot records.

    Gate fields use available physical controls and platform inputs; no
    record/sample/experiment identity or historical split is emitted.
    No outcome lookup or external file lookup is performed.
    """
    native = isinstance(original_row.get("experiment"), Mapping)
    inputs, code = _native_inputs(original_row) if native else _merged_inputs(original_row)
    complete = (all(inputs.get(name) is not None for name in CONTROL_COLUMNS)
                and all(inputs.get(f"post_{solvent}_volpct") is not None
                        for solvent in ("methanol", "ethanol", "water")))
    if native:
        experiment = original_row["experiment"]
        blade = any(step.get("kind") == "deposit" and step.get("deposition_mode") == "blade"
                    for step in experiment.get("steps", []))
        grade = any(re.search(r"ph\s*1000", str(material.get("grade", {}).get("value", "")), re.I)
                    for material in experiment.get("materials", []))
    else:
        blade = bool(re.search(r"blade", str(original_row.get("deposition_method", "")), re.I))
        grade = bool(re.search(r"ph\s*1000", str(original_row.get("polymer_grade", "")), re.I))
    out: dict[str, float | str] = {"pb_controls_complete": float(complete),
        "pb_platform_supported": float(complete and blade and grade)}
    if not inputs:
        return out
    for name in CORE_KEYS:
        _put(out, name, inputs.get(name))
    out["pb_recipe_additive_volpct_basis"] = "stock_or_mixture_denominator_unresolved"
    out["pb_post_volpct_basis"] = "total_post_solvent_mixture"
    if code in POST_SOLVENT_RECIPES:
        out["pb_post_solvent_category"] = "mixture_" + code
    else:
        out["pb_post_solvent_category"] = "unknown"
    for candidate in POST_SOLVENT_RECIPES:
        out["pb_post_solvent_is_" + candidate] = float(candidate == code)
    dmso, eg = inputs.get("dmso_volpct_reported"), inputs.get("eg_volpct_reported")
    if dmso is not None and eg is not None:
        _put(out, "reported_additive_volpct_sum", dmso + eg)
        if dmso + eg > 0:
            _put(out, "dmso_share_of_reported_additive_volpct", dmso / (dmso + eg))
        else:
            _put(out, "dmso_share_of_reported_additive_volpct", None)
    for stage, speed_name, temperature_name in (
        ("first_coating", "first_coating_speed_mm_s", "first_coating_stage_temperature_C"),
        ("post", "post_speed_mm_s", "post_stage_temperature_C"),
    ):
        speed = inputs.get(speed_name)
        if speed is not None and speed > 0:
            _put(out, stage + "_log_speed_mm_s", math.log(speed))
            _put(out, stage + "_inverse_speed_s_mm", 1 / speed)
        temperature = inputs.get(temperature_name)
        if temperature is not None:
            _put(out, stage + "_stage_temperature_K", temperature + 273.15)
    composition = [inputs.get(f"post_{name}_volpct") for name in ("methanol", "ethanol", "water")]
    if all(value is not None for value in composition):
        _put(out, "post_component_volpct_sum", sum(composition))
        out["pb_post_mixture_sum_is_100"] = float(math.isclose(sum(composition), 100.))
        if code in POST_SOLVENT_RECIPES:
            out["pb_post_category_composition_consistent"] = float(tuple(composition) == POST_SOLVENT_RECIPES[code])
    return out
