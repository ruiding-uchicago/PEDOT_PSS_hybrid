"""Conservative, target-free process scalars layered over frozen features.

Only post_treatment, process_sequence and measurement_method are read. Values
retain their processing stage, chemical object and explicit unit. A solution
concentration is not a delivered film dose. Conflicting scalars are omitted.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from typing import Any, Mapping

PROCESS_FEATURE_VERSION = "explicit-process-v1"
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
CONCENTRATION = re.compile(
    rf"(?<![\w.])(?P<value>{NUMBER})\s*(?P<unit>"
    r"(?:wt|weight|mass)\.?\s*%|(?:vol|volume)\.?\s*%|"
    r"%\s*\(?\s*(?:v/v|w/w)\s*\)?|"
    r"mmol\s*(?:/\s*l|l\s*\^?\s*-1)|"
    r"mol\s*(?:/\s*l|l\s*\^?\s*-1)|mM\b|M\b|N\b|%)", re.I)
CHEMICAL_PATTERNS = {
    "sulfuric_acid": r"sul(?:f|ph)uric\s+acid|h2so4",
    "hydrochloric_acid": r"hydrochloric\s+acid|\bhcl\b",
    "nitric_acid": r"nitric\s+acid|hno3",
    "phosphoric_acid": r"phosphoric\s+acid|h3po4",
    "formic_acid": r"formic\s+acid|hcooh",
    "acetic_acid": r"acetic\s+acid|ch3cooh",
    "methanesulfonic_acid": r"methanesulfonic\s+acid|\bmsa\b",
    "p_toluenesulfonic_acid": r"toluenesulfonic\s+acid|\bptsa\b",
    "hydroiodic_acid": r"hydroiodic\s+acid|\bhi\b",
    "mai": r"methylammonium\s+iodide|\bmai\b",
    "mabr": r"methylammonium\s+bromide|\bmabr\b",
    "macl": r"methylammonium\s+chloride|\bmacl\b",
    "nai": r"sodium\s+iodide|\bnai\b",
    "sodium_hydroxide": r"sodium\s+hydroxide|\bnaoh\b",
    "dmso": r"dimethyl\s+sulfoxide|\bdmso\b",
    "eg": r"(?<!di)(?<!tri)(?<!poly)ethylene\s+glycol|\beg\b",
    "methanol": r"\bmethanol\b|\bmeoh\b",
    "ethanol": r"\bethanol\b|\betoh\b",
    "ipa": r"isopropanol|isopropyl\s+alcohol|\bipa\b",
    "dmf": r"dimethylformamide|\bdmf\b",
    "gbl": r"gamma.butyrolactone|\bgbl\b",
    "water": r"\bwater\b|\bh2o\b",
    "emitfsi": r"\bemitfsi\b|\bemim.?tfsi\b",
}
ACIDS = frozenset(name for name in CHEMICAL_PATTERNS if name.endswith("_acid"))
DEPOSITION = re.compile(
    r"spin[ -]?coat|blade[ -]?coat|first blade|drop[ -]?cast|\bcast(?:ing)?\b|"
    r"screen[ -]?print|ink[ -]?jet|spray[ -]?coat|slot[ -]?die|wet[ -]?spinn", re.I)


def _text(value: Any) -> str:
    return "" if value is None else unicodedata.normalize("NFKC", str(value)).replace("−", "-")


def _valid_scalar(text: str, match: re.Match) -> bool:
    """Reject endpoints, inequalities, approximated ranges and negative values."""
    before = text[max(0, match.start() - 18):match.start()]
    after = text[match.end():match.end() + 18]
    if re.search(r"[<>≤≥]\s*$|\d\s*(?:[-–—]|to)\s*$|(?<!\w)-\s*$", before, re.I):
        return False
    if re.match(r"\s*(?:[-–—]|to)\s*\d", after, re.I):
        return False
    return True


def _unit(unit: str) -> tuple[str, float]:
    compact = re.sub(r"[\s().]", "", unit)
    lower = compact.lower()
    if "wt" in lower or "weight" in lower or "mass" in lower or "w/w" in lower:
        return "wtpct", 1.0
    if "vol" in lower or "v/v" in lower:
        return "volpct", 1.0
    # Unit case matters: M is molarity and mM is millimolarity.
    if compact == "mM" or lower.startswith("mmol"):
        return "mol_L", .001
    if compact == "M" or lower.startswith("mol"):
        return "mol_L", 1.0
    if compact == "N":
        return "normality_eq_L", 1.0
    return "pct_basis_unresolved", 1.0


def _clauses(row: Mapping) -> list[tuple[str, str]]:
    clauses: list[tuple[str, str]] = []
    post = _text(row.get("post_treatment"))
    clauses.extend(("post", part) for part in re.split(r";|→|->", post) if part.strip())
    sequence = _text(row.get("process_sequence"))
    parts = [part for part in re.split(r";|→|->", sequence) if part.strip()]
    boundary = next((i for i, part in enumerate(parts) if DEPOSITION.search(part)), None)
    for i, part in enumerate(parts):
        if boundary is not None:
            stage = "pre_film" if i < boundary else "deposition" if i == boundary else "after_deposition"
        elif re.search(r"post[ -]?(?:treat|process)|existing\s+film|film\s+(?:was\s+)?(?:immers|soak)", part, re.I):
            stage = "post"
        else:
            stage = "unspecified_stage"
        clauses.append((stage, part))
    method = _text(row.get("measurement_method"))
    clauses.extend(("measurement", part) for part in re.split(r";|→|->", method) if part.strip())
    return clauses


def enrich_process_features(row: Mapping[str, Any],
                            base_features: Mapping[str, float | str]) -> dict[str, float | str]:
    """Return a copy of base_features enriched with explicitly reported proc_* scalars.

    No target, sample/series label, locator, review or identifier is accessed.
    An unknown value stays absent with has_proc_* = 0. No absent value becomes 0.
    """
    out = dict(base_features)
    values: dict[str, list[float]] = defaultdict(list)
    flags: dict[str, set[float]] = defaultdict(set)

    def add(key: str, value: float) -> None:
        if math.isfinite(value):
            values[f"proc_{key}"].append(float(value))

    stages = ("pre_film", "deposition", "after_deposition", "post", "unspecified_stage", "measurement")
    for stage in stages:
        for name in ("acid_solution_concentration_mol_L", "acid_solution_concentration_wtpct",
                     "acid_solution_concentration_volpct", "treatment_cycles", "laser_power_W",
                     "laser_power_density_W_cm2", "uv_wavelength_nm", "uv_on"):
            out[f"has_proc_{stage}_{name}"] = 0.0

    for stage, clause in _clauses(row):
        # Each quantity is associated within one clause. No row-wide nearest
        # number matching or raw narrative categorical feature is emitted.
        chemicals = [(name, match) for name, pattern in CHEMICAL_PATTERNS.items()
                     for match in re.finditer(pattern, clause, re.I)]
        for match in CONCENTRATION.finditer(clause):
            if not _valid_scalar(clause, match):
                continue
            number = float(match.group("value"))
            unit, factor = _unit(match.group("unit"))
            choices = []
            for name, chemical_match in chemicals:
                distance = max(chemical_match.start() - match.end(), match.start() - chemical_match.end(), 0)
                between = (clause[match.end():chemical_match.start()] if chemical_match.start() >= match.end()
                           else clause[chemical_match.end():match.start()])
                # Do not attach an unrecognized solute's molarity to its named
                # solvent (e.g. 0.1 M unknown-salt/DMF). Intervening chemical
                # words are not connectors, so association remains missing.
                connectors = {"of", "in", "at", "aqueous", "solution", "bath",
                    "concentration", "about", "approximately", "by", "volume", "weight", "mass"}
                words = set(re.findall(r"[a-z]{2,}", between.lower()))
                if (distance <= 28 and words <= connectors
                        and not re.search(r"[,;]|\b(?:then|and|subsequently)\b", between, re.I)):
                    choices.append((distance, name))
            choices.sort()
            if not choices or (len(choices) > 1 and choices[0][0] == choices[1][0] and choices[0][1] != choices[1][1]):
                continue
            chemical = choices[0][1]
            add(f"{stage}_{chemical}_solution_concentration_{unit}", number * factor)
            if chemical in ACIDS:
                add(f"{stage}_acid_solution_concentration_{unit}", number * factor)

        # Count only repetitions explicitly linked to a treatment/rinse/wash.
        count_patterns = (
            r"(?:treat(?:ed|ment)?|rins(?:e|ed|ing)|wash(?:ed|ing)?|immers(?:ed|ion)|soak(?:ed|ing)?)"
            r"[^;,.]{0,35}?\b(?P<count>\d+|once|twice|thrice|one|two|three|four|five)\s*(?:times|cycles|repetitions)\b",
            r"\b(?P<count>once|twice|thrice)\s*(?:[- ]treated|treat(?:ed|ment)?)\b",
            r"\b(?:treat(?:ed|ment)?)\s+(?P<count>once|twice|thrice)\b",
        )
        word_counts = {"once": 1, "one": 1, "twice": 2, "two": 2, "thrice": 3,
                       "three": 3, "four": 4, "five": 5}
        for pattern in count_patterns:
            for match in re.finditer(pattern, clause, re.I):
                token = match.group("count").lower()
                count = word_counts.get(token, int(token) if token.isdigit() else None)
                if count is not None and count > 0:
                    operation = "rinse_cycles" if re.search(r"rins", match.group(0), re.I) else (
                        "wash_cycles" if re.search(r"wash", match.group(0), re.I) else "treatment_cycles")
                    add(f"{stage}_{operation}", count)

        laser = bool(re.search(r"\blaser\b|nd:yag|photonic", clause, re.I))
        uv = bool(re.search(r"\buv\b|ultraviolet", clause, re.I))
        if laser or uv:
            for match in re.finditer(rf"(?<![\w.])(?P<value>{NUMBER})\s*(?P<unit>mW|W)\s*(?P<density>(?:/\s*cm\s*\^?\s*2|cm\s*\^?\s*-2))?", clause):
                if not _valid_scalar(clause, match):
                    continue
                power = float(match.group("value")) * (.001 if match.group("unit") == "mW" else 1)
                kind = "power_density_W_cm2" if match.group("density") else "power_W"
                if laser:
                    add(f"{stage}_laser_{kind}", power)
                if uv:
                    add(f"{stage}_uv_{kind}", power)
            for match in re.finditer(rf"(?<![\w.])(?P<value>{NUMBER})\s*nm\b", clause):
                if _valid_scalar(clause, match):
                    if laser:
                        add(f"{stage}_laser_wavelength_nm", float(match.group("value")))
                    if uv:
                        add(f"{stage}_uv_wavelength_nm", float(match.group("value")))
        if uv:
            if re.search(r"\buv(?:\s+light|\s+state)?\s*(?:is\s*)?(?:off|disabled)\b|without\s+uv|no\s+uv", clause, re.I):
                flags[f"proc_{stage}_uv_on"].add(0.0)
            if re.search(r"\buv(?:\s+light|\s+state)?\s*(?:is\s*)?(?:on|enabled)\b|under\s+uv|with\s+(?:\d+\s*nm\s+)?uv", clause, re.I):
                flags[f"proc_{stage}_uv_on"].add(1.0)
        # Named laser controls have explicit units and are not temperatures or
        # powers selected by their numeric magnitude.
        if laser:
            for match in re.finditer(rf"(?<![\w.])(?P<value>{NUMBER})\s*kHz\b", clause, re.I):
                if _valid_scalar(clause, match):
                    add(f"{stage}_laser_frequency_Hz", float(match.group("value")) * 1000)
            speed = re.search(rf"(?P<value>{NUMBER})\s*mm\s*(?:/\s*s\b|s\s*\^?\s*-1)", clause, re.I)
            if speed and _valid_scalar(clause, speed):
                add(f"{stage}_laser_raster_speed_m_s", float(speed.group("value")) * .001)
        # Solution pH is accepted only when explicitly named in an allowed
        # process field; never inferred from acid molarity or a sample label.
        if stage != "measurement":
            for match in re.finditer(rf"\bpH\s*(?:=|≈|~|of|:)??\s*(?P<value>{NUMBER})(?![\w.])", clause, re.I):
                if _valid_scalar(clause, match):
                    ph = float(match.group("value"))
                    if 0 <= ph <= 14:
                        add(f"{stage}_solution_ph", ph)

    for key, candidates in values.items():
        unique = set(candidates)
        out["has_" + key] = float(len(unique) == 1)
        if len(unique) == 1:
            out[key] = next(iter(unique))
        else:
            out["ambiguous_" + key] = 1.0
    for key, candidates in flags.items():
        out["has_" + key] = float(len(candidates) == 1)
        if len(candidates) == 1:
            out[key] = next(iter(candidates))
        else:
            out["ambiguous_" + key] = 1.0
    return out
