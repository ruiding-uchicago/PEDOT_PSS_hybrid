"""Minimal physical/type validation of the public experimental input."""
from collections.abc import Mapping
import math
from numbers import Real

from ._vendor.clean_inputs import (INPUT_FIELDS, TEXT_FIELDS, NUMERIC_FIELDS,
    EXTRA_SCALAR_NAMES, STAGE_KINDS, STAGE_NUMERIC_FIELDS, STAGE_TEXT_FIELDS)

_DURATION_UNITS = {"s", "sec", "second", "seconds", "min", "minute", "minutes", "h", "hr", "hours", "day", "days", "week", "weeks"}
_TEMP_UNITS = {"c", "°c", "degc", "celsius"}
_SPEED_UNITS = {"mm/s", "mm/sec", "m/s", "m/min", "cm/s", "cm/min", "mm/min", "um/s"}
_LENGTH_UNITS = {"nm", "um", "mm", "cm", "m"}
_CONCENTRATION_UNITS = {"mol/l", "m", "mm", "mmol/l", "mol_l", "wt%", "wt.%", "vol%", "vol.%", "%", "mg/ml", "g/l", "g/ml", "w/v%", "v/v%", "w/w%"}
_DIMENSIONLESS = {None, "1", "dimensionless", "unitless", "ph", "ratio"}
_SECONDS_PER_UNIT = {"s": 1., "sec": 1., "second": 1., "seconds": 1.,
    "min": 60., "minute": 60., "minutes": 60., "h": 3600., "hr": 3600.,
    "hours": 3600., "day": 86400., "days": 86400., "week": 604800., "weeks": 604800.}
_MICROMETERS_PER_UNIT = {"nm": .001, "um": 1., "mm": 1000., "cm": 10000., "m": 1e6}


def _number(value, path, *, minimum=None, maximum=None, temperature=False):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number or null; numeric strings and booleans are not accepted")
    if temperature and value <= -273.15:
        raise ValueError(f"{path} must be greater than -273.15 °C")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must be at most {maximum}")


def _list(value, path):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list or null")
    return value


def _units(name):
    if name in {"ph", "solution_ph", "total_draw_ratio", "preparation_water_ph"}:
        return _DIMENSIONLESS
    if name.startswith("preparation_water_") and name.endswith("_mg_l"):
        return {"mg/l"}
    if name == "blend_pedot_pss_dry_mass_percent":
        return {"%", "wt%", "wt.%", "w/w%"}
    if "temperature" in name:
        return _TEMP_UNITS
    if "duration" in name or "time" in name:
        return _DURATION_UNITS
    if name in {"mix_rotation_speed", "spin_speed"}:
        return {"rpm", "r/min", "rev/min"}
    if name in {"flow_rate"}:
        return {"ml/min", "ul/min", "ml/s", "l/min"}
    if name == "pump_speed":
        return _SPEED_UNITS | {"rpm", "ml/min", "ul/min"}
    if "speed" in name:
        return _SPEED_UNITS
    if name in {"laser_line_gap", "nozzle_diameter", "nozzle_to_substrate_distance", "drop_spacing", "blade_gap", "planned_via_diameter"}:
        return _LENGTH_UNITS
    if name == "stock_volume":
        return {"ml", "ul", "l"}
    if name == "relative_humidity":
        return {"%", "%rh", "%rh", "rh%"}
    if name == "measurement_frequency":
        return {"hz", "khz", "mhz", "ghz"}
    if "voltage" in name or name == "measurement_bias":
        return {"v", "mv", "kv"}
    if name in {"final_ink_solids_wt_percent"}:
        return {"wt%", "wt.%", "%", "w/w%"}
    if "concentration" in name or name == "organic_solvent_content":
        return _CONCENTRATION_UNITS
    # All public allowlisted names must have a declared dimensional rule.
    raise ValueError(f"No supported unit rule for extra scalar {name}")


def validate(experiment):
    if not isinstance(experiment, Mapping):
        raise TypeError("An experiment must be a dictionary of experimental conditions")
    if any(not isinstance(key, str) for key in experiment):
        raise ValueError("Experimental input field names must be strings")
    unknown = sorted(set(experiment) - set(INPUT_FIELDS))
    if unknown:
        raise ValueError("Unknown experimental input fields: " + ", ".join(unknown))
    for field in TEXT_FIELDS:
        value = experiment.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
    for field in NUMERIC_FIELDS:
        _number(experiment.get(field), field,
            temperature=field.endswith("temperature_C"),
            minimum=0 if "duration" in field or field == "stock_solids_wt_percent" else None,
            maximum=100 if field == "stock_solids_wt_percent" else None)
    names = _list(experiment.get("additive_names"), "additive_names")
    amounts = _list(experiment.get("additive_amounts_raw"), "additive_amounts_raw")
    if len(names) != len(amounts):
        raise ValueError("additive_names and additive_amounts_raw must have equal lengths; use null for an unknown amount")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Each additive name must be a nonempty string")
    for i, amount in enumerate(amounts):
        if isinstance(amount, str) or amount is None:
            continue
        _number(amount, f"additive_amounts_raw[{i}]", minimum=0)
    for i, item in enumerate(_list(experiment.get("extra_scalar_features"), "extra_scalar_features")):
        path = f"extra_scalar_features[{i}]"
        if not isinstance(item, Mapping) or set(item) != {"name", "value", "unit"}:
            raise ValueError(f"{path} must contain name, value and unit, with no extra fields")
        name = item.get("name")
        if name not in EXTRA_SCALAR_NAMES:
            raise ValueError(f"{path}.name is not a supported experimental scalar")
        if item.get("value") is None:
            raise ValueError(f"{path}.value must be a finite number; omit unknown extra scalars")
        value = item["value"]
        temperature = "temperature" in name
        minimum = 0 if ("duration" in name or "time" in name or "speed" in name
            or name in {"relative_humidity", "final_ink_solids_wt_percent", "stock_volume", "blade_gap"}) else None
        maximum = 100 if name in {"relative_humidity", "final_ink_solids_wt_percent", "blend_pedot_pss_dry_mass_percent"} else None
        if name == "blend_pedot_pss_dry_mass_percent" or name.startswith("preparation_water_") and name.endswith("_mg_l"):
            minimum = 0
        _number(value, path + ".value", minimum=minimum, maximum=maximum, temperature=temperature)
        if name == "planned_via_diameter" and value <= 0:
            raise ValueError(f"{path}.value must be positive for planned_via_diameter")
        unit = item.get("unit")
        if unit is not None and not isinstance(unit, str):
            raise ValueError(f"{path}.unit must be a unit string")
        normalized_unit = unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "") if unit is not None else None
        if normalized_unit not in _units(name):
            raise ValueError(f"{path}.unit {unit!r} is not supported for {name}")
    allowed_stage_fields = set(STAGE_TEXT_FIELDS) | set(STAGE_NUMERIC_FIELDS) | {"temperature_range_C", "duration_range_s"}
    for i, stage in enumerate(_list(experiment.get("process_stages"), "process_stages")):
        path = f"process_stages[{i}]"
        if not isinstance(stage, Mapping) or set(stage) - allowed_stage_fields:
            raise ValueError(f"{path} has unsupported stage fields; temperatures use temperature_C and durations duration_s")
        if stage.get("kind") not in STAGE_KINDS:
            raise ValueError(f"{path}.kind must name a supported process stage")
        for field in STAGE_TEXT_FIELDS:
            value = stage.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{path}.{field} must be a string or null")
        for field in STAGE_NUMERIC_FIELDS:
            _number(stage.get(field), f"{path}.{field}", temperature=field == "temperature_C",
                minimum=None if field in {"temperature_C", "pressure_bar", "voltage_V"} else 0,
                maximum=100 if field == "relative_humidity_percent" else None)
        for field in ("temperature_range_C", "duration_range_s"):
            bounds = stage.get(field)
            if bounds is None:
                continue
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"{path}.{field} must contain two numeric bounds or null")
            for j, value in enumerate(bounds):
                if value is None:
                    raise ValueError(f"{path}.{field} cannot contain null bounds")
                _number(value, f"{path}.{field}[{j}]", temperature=field == "temperature_range_C",
                    minimum=0 if field == "duration_range_s" else None)
            if bounds[0] > bounds[1]:
                raise ValueError(f"{path}.{field} bounds must be in increasing order")


def canonical_extra_units(experiment):
    """Convert public unit aliases to the explicit new-condition training units.

    This boundary conversion leaves the frozen scientific feature builder and
    caller dictionary unchanged. Unrelated experimental units retain their
    original representation.
    """
    result = dict(experiment)
    if experiment.get("extra_scalar_features") is None:
        return result
    extras = []
    for item in experiment["extra_scalar_features"]:
        item = dict(item)
        name, value, unit = item["name"], item["value"], item["unit"]
        unit_key = unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "") if unit is not None else None
        if name == "planned_via_diameter":
            value = float(value) * _MICROMETERS_PER_UNIT[unit_key]
            unit = "um"
        elif name == "measurement_bias":
            value = float(value) * {"mv": .001, "v": 1., "kv": 1000.}[unit_key]
            unit = "V"
        elif name == "blend_pedot_pss_dry_mass_percent":
            unit = "%"
        elif name.startswith("preparation_water_") and name.endswith("_mg_l"):
            unit = "mg/L"
        elif name == "preparation_water_ph":
            unit = "dimensionless"
        elif name == "blade_gap":
            value = float(value) * _MICROMETERS_PER_UNIT[unit_key]
            unit = "um"
        elif name == "stock_volume":
            value = float(value) * {"ul": .001, "ml": 1., "l": 1000.}[unit_key]
            unit = "mL"
        elif name in {"mix_duration", "substrate_uvo_duration", "post_dry_duration",
                      "second_anneal_duration", "air_storage_time"}:
            value = float(value) * _SECONDS_PER_UNIT[unit_key]
            unit = "s"
        elif name == "mix_rotation_speed":
            unit = "rpm"
        elif "temperature" in name:
            unit = "C"
        _number(value, f"extra_scalar_features[{len(extras)}].value")
        item.update(value=value, unit=unit)
        extras.append(item)
    result["extra_scalar_features"] = extras
    return result


def require_condition(clean):
    unknown = {"unknown", "unreported", "not reported", "not_reported", "n/a", "na", "missing", "unresolved"}
    for value in clean.values():
        if value is None or value == [] or value == "":
            continue
        if isinstance(value, str) and value.strip().lower() in unknown:
            continue
        return
    raise ValueError("Supply at least one known planned experimental condition; empty/all-null input cannot identify an experiment")
