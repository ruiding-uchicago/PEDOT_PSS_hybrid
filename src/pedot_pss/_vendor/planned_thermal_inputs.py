"""Original deterministic thermal-stage preparation; no model fitting or IO."""
from __future__ import annotations
import math
import re
import numpy as np

# Remove every historical time alias when an explicit ordered stage list is
# authoritative. Temperature-only deposition contexts may remain: without a
# hold duration they cannot accidentally become another thermal exposure.
_LEGACY_THERMAL_TIMES = (
    "dry_duration_s", "dry_duration_min", "anneal_duration_s", "anneal_duration_min",
    "post_treatment_duration_s", "post_treatment_duration_min",
    "pb_first_dry_duration_s", "pb_first_anneal_duration_s",
    "pb_post_dry_duration_s", "pb_second_anneal_duration_s",
    "extra_post_dry_duration__s", "extra_second_anneal_duration__s",
)
_THERMAL_STAGE_KINDS = frozenset({"dry", "anneal", "heat", "press"})


def stage_thermal_budget(stages):
    """Equivalent minutes at 130 C from supported constant thermal holds.

    Only the explicitly ordered dry/anneal/heat/press holds enter this clock.
    A cooling endpoint, ramp, temperature interval or missing duration is not
    turned into a constant-temperature hold. Omitted history stays flagged;
    it is never described as a measured zero exposure.
    """
    budgets, unknown, intervals = [], 0, 0
    for stage in stages:
        if stage.get("kind") not in _THERMAL_STAGE_KINDS:
            continue
        if stage.get("temperature_range_C") or stage.get("duration_range_s"):
            intervals += 1
            unknown += 1
            continue
        method = str(stage.get("method") or "").lower()
        if re.search(r"\bramp\b|temperature\s+sweep|(?:heating|cooling)\s+rate", method):
            # An endpoint and total ramp duration do not specify its thermal
            # trajectory, and are not a constant-temperature exposure.
            unknown += 1
            continue
        temp, duration = stage.get("temperature_C"), stage.get("duration_s")
        if temp is None or duration is None or temp <= -273.15 or duration < 0:
            unknown += 1
            continue
        if duration == 0:
            budgets.append(0.)
            continue
        log_budget = math.log(duration / 60.) - .35 / 8.617333262e-5 * (
            1. / (temp + 273.15) - 1. / 403.15)
        # Match the declared numerical exposure bounds of the legacy clock.
        budgets.append(math.exp(float(np.clip(log_budget, -60., 30.))))
    return {
        "equivalent_minutes_130C": math.fsum(budgets) if budgets else None,
        "supported_stage_count": len(budgets), "unknown_thermal_stage_count": unknown,
        "interval_thermal_stage_count": intervals,
    }


def physics_rows_for_stages(clean, scientific):
    """Prepare authoritative thermal rows without duplicating flat clocks.

    Historical flat fields remain in ML's original scientific view. The
    physical view instead contains exactly one equivalent annealing exposure
    representing the supported ordered holds; legacy stage durations are
    removed only from this separate copy.
    """
    rows = []
    for canonical, original in zip(clean, scientific):
        row = dict(original)
        stages = canonical.get("process_stages") or []
        if stages:
            clock = stage_thermal_budget(stages)
            for key in _LEGACY_THERMAL_TIMES:
                row.pop(key, None)
            # _prepare_rows sees only the aggregate, not flat-field duplicates.
            row.pop("anneal_temperature_C", None)
            row.pop("pb_first_anneal_temperature_C", None)
            if clock["equivalent_minutes_130C"] is not None:
                row["anneal_temperature_C"] = 130.
                row["anneal_duration_s"] = clock["equivalent_minutes_130C"] * 60.
            for name, value in clock.items():
                original["planned_history_" + name] = value
            original["planned_history_partial"] = float(clock["unknown_thermal_stage_count"] > 0)
        rows.append(row)
    return rows

