"""Outcome-free recipe features for the expanded PEDOT:PSS intake.

The default mode uses formulation, processing and measurement conditions.
Observed-film characterizations and source identity are opt-in ablations.
Unknown numeric values are omitted and accompanied by explicit ``has_*`` flags.
Molecular summaries describe the listed molecules; they are NOT effective
properties of a mixture or of the processed PEDOT:PSS film.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

FEATURE_VERSION = "expanded-recipe-v1"
DESCRIPTOR_PATH = Path(__file__).with_name("molecular_descriptors.json")

# A closed chemical vocabulary prevents arbitrary narratives/identifiers from
# becoming high-cardinality recipe tokens. Commercial products are categorical
# materials only; no small-molecule descriptors are assigned to them.
CHEMICAL_ALIASES = {
    "dmso": (r"dimethyl\s*sul(?:f|ph)oxide", r"\bdmso\b"),
    "eg": (r"(?<!di)(?<!tri)(?<!poly)ethylene\s+glycol", r"\beg\b"),
    "deg": (r"diethylene\s+glycol", r"\bdeg\b"),
    "teg": (r"triethylene\s+glycol", r"\bteg\b"),
    "dmf": (r"dimethylformamide", r"\bdmf\b"),
    "nmp": (r"methyl.{0,3}pyrrolid(?:one|inone)", r"\bnmp\b"),
    "methanol": (r"\bmethanol\b", r"\bmeoh\b"),
    "ethanol": (r"\bethanol\b", r"\betoh\b"),
    "ipa": (r"isoprop(?:yl\s+alcohol|anol)", r"2.propanol", r"\bipa\b"),
    "water": (r"\bwater\b", r"\bh2o\b"),
    "acetone": (r"\bacetone\b",),
    "acetonitrile": (r"\bacetonitrile\b", r"\bmecn\b"),
    "chlorobenzene": (r"\bchlorobenzene\b",),
    "formamide": (r"(?<!dimethyl)\bformamide\b",),
    "glycerol": (r"\bglycer(?:ol|in)\b",),
    "methoxyethanol": (r"methoxyethanol", r"\b2.me\b"),
    "thf": (r"tetrahydrofuran", r"\bthf\b"),
    "toluene": (r"\btoluene\b",),
    "sulfuric_acid": (r"sul(?:f|ph)uric\s+acid", r"h2so4"),
    "hydrochloric_acid": (r"hydrochloric\s+acid", r"\bhcl\b"),
    "nitric_acid": (r"nitric\s+acid", r"hno3"),
    "phosphoric_acid": (r"phosphoric\s+acid", r"h3po4"),
    "formic_acid": (r"formic\s+acid", r"hcooh"),
    "acetic_acid": (r"acetic\s+acid", r"ch3cooh"),
    "methanesulfonic_acid": (r"methane\s*sulfonic\s+acid", r"\bmsa\b"),
    "p_toluenesulfonic_acid": (r"toluenesulfonic\s+acid", r"\bptsa\b", r"\btsa\b"),
    "oxalic_acid": (r"oxalic\s+acid",),
    "sorbitol": (r"\bsorbitol\b",),
    "gops": (r"glycidoxypropyl", r"\bgops\b",),
    "triton_x100": (r"triton\s*x.?100",),
    "zonyl": (r"\bzonyl\b",),
    "dynol": (r"\bdynol\b",),
    "sds": (r"sodium\s+dodecyl\s*sulfate", r"\bsds\b"),
    "sdbs": (r"dodecylbenzene\s*sulfonate", r"\bsdbs\b"),
    "peg": (r"polyethylene\s+glycol", r"\bpeg\b",),
    "pva": (r"polyvinyl\s+alcohol", r"\bpva\b",),
    "pvp": (r"polyvinylpyrrolidone", r"\bpvp\b",),
    "pei": (r"polyethylenimine", r"\bpei\b",),
    "cnt": (r"carbon\s+nanotube", r"\b(?:mwcnt|swcnt|cnt)s?\b"),
    "graphene": (r"graphene", r"\brgo\b",),
    "silver": (r"\bsilver\b", r"\bag\b",),
    "gold": (r"\bgold\b", r"\bau\b",),
    "hydrazine": (r"\bhydrazine\b", r"\bhz\b",),
    "urea": (r"\burea\b",),
    "tmu": (r"tetramethylurea", r"\btmu\b",),
    "tdae": (r"tetrakis.*amino.*ethylene", r"\btdae\b",),
}
ACIDS = frozenset(k for k in CHEMICAL_ALIASES if k.endswith("_acid"))
POLAR_SOLVENTS = frozenset({"dmso", "eg", "deg", "teg", "dmf", "nmp", "methanol",
    "ethanol", "ipa", "water", "acetone", "acetonitrile", "formamide", "glycerol",
    "methoxyethanol", "thf"})
COMMERCIAL_POLYMERS = frozenset({"triton_x100", "zonyl", "dynol", "peg", "pva", "pvp",
    "pei", "cnt", "graphene", "silver", "gold"})

NUMERIC_FIELDS = (
    "stock_solids_wt_percent", "dry_temperature_C", "dry_duration_min",
    "anneal_temperature_C", "anneal_duration_min", "post_treatment_temperature_C",
    "post_treatment_duration_min", "measurement_temperature_C",
)
OBSERVED_FIELDS = (
    "thickness_nm", "coverage_percent", "crystallinity_percent", "lattice_spacing_A",
    "coherence_length_nm", "raman_peak_cm_inv", "xps_pss_pedot_ratio",
)
# This explicit allowlist excludes fitted hopping T0, electrical transport,
# resistivity/sheet resistance, power factor and other conductivity proxies.
RECIPE_EXTRA_NAMES = frozenset({
    "coating_speed", "coating_stage_temperature", "post_coating_speed",
    "post_coating_stage_temperature", "post_dry_duration", "second_anneal_temperature",
    "second_anneal_duration", "spin_speed", "spin_duration", "pump_speed",
    "writing_speed", "motion_platform_speed", "laser_line_gap", "total_draw_ratio",
    "ph", "solution_ph", "concentration", "hz_concentration", "tdae_concentration",
    "tmu_concentration", "final_ink_solids_wt_percent", "working_solution_concentration",
    "organic_solvent_content", "pss_polymerization_time", "ultrasonication_time",
    "water_immersion_duration", "water_immersion_time", "treating_temperature",
    "vacuum_drying_duration", "post_wash_drying_temperature", "post_wash_drying_duration",
    "air_storage_time", "relative_humidity", "measurement_frequency",
    "applied_voltage", "ebl_acceleration_voltage", "tensile_strain",
    "maximum_strain_on_pdms", "maximum_strain_on_prestrained_pdms",
})
OBSERVED_EXTRA_NAMES = frozenset({"dielectric_constant", "dipole_moment",
    "surface_tension", "density", "cv_scan_rate"})
OBSERVED_EXTRA_PATTERN = re.compile(
    r"thickness|roughness|^afm|^rq$|^ra$|crystalli|^xps|^raman|^giwaxs|"
    r"lattice|coherence|stacking|^d100|work_function|contact_angle|^d_spacing|"
    r"^grain_size|^structure_factor|^form_factor|^edge_on|^pedot_to_pss_ratio", re.I)
FORBIDDEN_EXTRA_PATTERN = re.compile(
    r"conductiv|resist|sigma|power.?factor|seebeck|carrier|mobility|\bt0\b|"
    r"vrh|activation_energy|replicat|sample|experiment|row_id|^index$|^id$", re.I)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).lower().strip().replace("−", "-")


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _text(value)).strip("_")


def _number(value: Any) -> float | None:
    """Accept explicit scalar values/means, never intervals or narrative digits."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    s = _text(value).replace(",", "")
    match = re.fullmatch(r"(?:~|≈|ca\.?\s*|approximately\s*)?\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
        r"(?:\s*(?:±|\+/-)\s*\d+(?:\.\d*)?(?:e[+-]?\d+)?)?", s)
    return _number(float(match.group(1))) if match else None


def _put_numeric(out: dict, key: str, value: Any) -> None:
    number = _number(value)
    out["has_" + key] = float(number is not None)
    if number is not None:
        out[key] = number


def _chemicals(text: Any) -> list[str]:
    s = _text(text)
    return [key for key, aliases in CHEMICAL_ALIASES.items()
            if any(re.search(alias, s) for alias in aliases)]


def _grade(text: Any) -> str:
    s = _text(text)
    patterns = (("ph1000", r"ph\s*1000"), ("ph500", r"ph\s*500"),
        ("ai4083", r"ai\s*4083"), ("ai7580", r"ai\s*7580"),
        ("baytron_p", r"baytron\s*p"), ("sv3", r"\bsv\s*3\b"),
        ("icp1020", r"icp\s*1020"), ("clevios_p", r"clevios\s*p\b"))
    found = [key for key, pattern in patterns if re.search(pattern, s)]
    return "+".join(found) if found else ("other_reported" if s else "unknown")


def _categories(out: dict, key: str, text: Any, patterns: Mapping[str, str]) -> None:
    s = _text(text)
    found = [name for name, pattern in patterns.items() if re.search(pattern, s)]
    out[key] = "+".join(found) if found else ("other_reported" if s else "unknown")
    for name in patterns:
        out[f"{key}_{name}"] = float(name in found)
    out[f"has_{key}"] = float(bool(s))


def _dose_basis(text: Any) -> str:
    s = _text(text)
    if not s or s in {"none", "not reported", "unknown"}:
        return "unknown"
    if re.search(r"unresolved|unclear|not clarified|not specified", s):
        return "unresolved_denominator"
    if re.search(r"solid.film|dry.film|film wt|film mass", s):
        return "film_solids"
    if re.search(r"solute|polymer solids|pedot.*pss solids|dry polymer", s):
        return "polymer_solids"
    if re.search(r"total|final mixture|mixture|blend", s):
        return "mixture"
    if re.search(r"stock|as.purchased|dispersion|casting solution|ph1000 solution", s):
        return "stock_dispersion"
    if re.search(r"molar|mol/|mol l|\bmm\b", s):
        return "solution_molarity"
    return "other_reported"


def _parse_dose(value: Any, basis: Any = None) -> tuple[float | None, str]:
    """Return a reported scalar and explicit unit; never infer wt% -> vol%."""
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        s = str(value)
    else:
        s = _text(value)
    if not s:
        return None, "unknown"
    # A range, inequality or multiple alternatives is not a scalar dose.
    if re.search(r"[<>≤≥]|\d\s*[-–—]\s*\d|\d\s+to\s+\d", s):
        return None, "interval_or_bound"
    patterns = (
        (r"([\d.]+)\s*(?:vol(?:ume)?\.?\s*%|%\s*\(?v/v\)?|v/v\s*%)", "volpct"),
        (r"([\d.]+)\s*(?:wt\.?\s*%|weight\s*%|mass\s*%|%\s*\(?w/w\)?)", "wtpct"),
        (r"([\d.]+)\s*(?:mol\.?\s*%|mole\s*%)", "molpct"),
        (r"([\d.]+)\s*(?:mmol\s*(?:/\s*l|l\s*\^?\s*-1)|mm\b)", "mmol_L"),
        (r"([\d.]+)\s*(?:mol\s*(?:/\s*l|l\s*\^?\s*-1)|m\b)", "mol_L"),
        (r"([\d.]+)\s*mg\s*/\s*ml", "mg_mL"),
        (r"([\d.]+)\s*ml\b", "mL"),
        (r"([\d.]+)\s*(?:ul|μl|µl)\b", "uL"),
        (r"([\d.]+)\s*mg\b", "mg"),
    )
    for pattern, unit in patterns:
        matches = list(re.finditer(pattern, s))
        if matches:
            values = {float(m.group(1)) for m in matches}
            if len(values) != 1:
                return None, "ambiguous_multiple_values"
            return next(iter(values)), unit
    # Percent without a declared unit remains its own coordinate. A basis may
    # supply a unit only when explicit; it never supplies a denominator value.
    match = re.fullmatch(r"\s*([\d.]+)\s*%?\s*", s)
    if not match:
        match = re.search(r"\b([\d.]+)\s*%(?!\w)", s)
    if match:
        b = _text(basis)
        unit = ("volpct" if re.search(r"vol(?:ume)?\.?\s*%|v/v", b) else
                "wtpct" if re.search(r"wt\.?\s*%|weight fraction|mass fraction|w/w", b) else
                "pct_unresolved" if "%" in s else "amount_unresolved")
        return float(match.group(1)), unit
    # Explicit component:stock ratios preserve direction and are not converted
    # to mixture fractions. Both ratio units and denominator are retained.
    match = re.fullmatch(r"\s*([\d.]+)\s*[:/]\s*([\d.]+)\s*(?:\(?v/v\)?)?\s*", s)
    if match and float(match.group(2)) > 0:
        return float(match.group(1)) / float(match.group(2)), "reported_ratio"
    return None, "unparsed"


def _ratio_features(out: dict, raw: Any) -> None:
    s = _text(raw)
    out["has_reported_pss_pedot_mass_ratio"] = 0.0
    out["has_reported_pss_pedot_molar_ratio"] = 0.0
    out["reported_ratio_direction"] = "unresolved"
    if not s:
        return
    if re.search(r"mwcnt|ag:|des|surfactant|solution:|emulsion|feed|edot|not clarified", s.replace("pedot", "polymer")):
        return
    if re.search(r"repeat|monomer|molar", s):
        unit = "molar"
    elif re.search(r"weight|mass|w/w|\bwt\b", s):
        unit = "mass"
    else:
        unit = "basis_unreported"
    reverse = bool(re.search(r"pss\s*(?::|/|-to-|to)\s*pedot", s))
    forward = bool(re.search(r"pedot\s*(?::|/|-to-|to)\s*pss", s))
    # Bare a:b follows the named input field order PEDOT:PSS. We expose that
    # interpretation but require explicit mass basis for the physics mass key.
    pair = re.search(r"([\d.]+)\s*[:/]\s*([\d.]+)", s)
    if pair:
        a, b = map(float, pair.groups())
        if min(a, b) <= 0:
            return
        ratio = a / b if reverse else b / a
        direction = "pss_to_pedot_explicit" if reverse else (
            "pedot_to_pss_explicit" if forward else "pedot_to_pss_field_order")
    else:
        if not (reverse or forward):
            return
        nums = re.findall(r"(?<![a-z\d])\d+(?:\.\d+)?", s)
        if len(nums) != 1 or float(nums[0]) <= 0:
            return
        ratio = float(nums[0]) if reverse else 1.0 / float(nums[0])
        direction = "pss_to_pedot_explicit" if reverse else "pedot_to_pss_explicit"
    out["reported_ratio_direction"] = direction
    out["reported_ratio_basis"] = unit
    out[f"reported_pss_pedot_{unit}_ratio"] = ratio
    out[f"has_reported_pss_pedot_{unit}_ratio"] = 1.0


OPERATION_PATTERNS = {
    "mix": r"mix|stirr|blend|sonicat", "filter": r"filter", "dry": r"dry|dried|drying",
    "anneal": r"anneal", "heat": r"heat|bake|thermal", "immerse": r"immers|dip|soak",
    "rinse": r"rins|wash", "vapor": r"vapou?r", "plasma": r"plasma",
    "uv_ozone": r"uv.{0,4}ozone|uvo\b", "stretch": r"stretch|drawn|drawing",
    "laser": r"laser|photonic|irradiat", "vacuum": r"vacuum", "press": r"press|calender",
    "acid": r"acid|h2so4|hcl|hno3", "solvent": r"solvent",
    "blade": r"blade", "spin": r"spin.coat", "cast": r"cast", "print": r"print",
}


def _process_features(out: dict, row: Mapping) -> None:
    post = _text(row.get("post_treatment"))
    sequence = _text(row.get("process_sequence"))
    negated_post = bool(re.fullmatch(r"(?:none|no(?:ne)?(?: post[- ]treatment)?|untreated|not applicable|n/a)", post))
    post_text = "" if negated_post else post
    out["has_post_treatment"] = float(bool(post_text))
    for name, pattern in OPERATION_PATTERNS.items():
        out[f"post_op_{name}"] = float(bool(re.search(pattern, post_text)))
    # Stage assignment uses explicit arrows/semicolons and the first film
    # deposition operation. Characterization/target phrases emit no tokens.
    steps = [s for s in re.split(r"→|->|;", sequence) if s.strip()]
    deposition = re.compile(r"spin.coat|blade.coat|drop.cast|casting|screen.print|inkjet|wet.spin|first blade")
    boundary = next((i for i, step in enumerate(steps) if deposition.search(step)), None)
    pre = " ".join(steps[:boundary]) if boundary is not None else ""
    after = " ".join(steps[boundary + 1:]) if boundary is not None else ""
    out["process_stage_boundary_known"] = float(boundary is not None)
    out["has_process_sequence"] = float(bool(sequence))
    for name, pattern in OPERATION_PATTERNS.items():
        out[f"pre_film_op_{name}"] = float(bool(re.search(pattern, pre)))
        out[f"after_deposition_op_{name}"] = float(bool(re.search(pattern, after)))
    post_chems = set(_chemicals(post_text))
    for chem in tuple(post_chems):
        # Honor explicit zero concentration even in solvent mixture text.
        for alias in CHEMICAL_ALIASES[chem]:
            if re.search(r"(?:" + alias + r")\s*[:=]?\s*0(?:\.0+)?\s*(?:vol\.?\s*%|wt\.?\s*%|%)", post_text):
                post_chems.discard(chem)
    for chem in CHEMICAL_ALIASES:
        out[f"post_{chem}_present"] = float(chem in post_chems)
    generic_acid = (bool(re.search(r"\bacid\b", post_text))
                    and not (set(_chemicals(post_text)) & ACIDS))
    out["acid_post_present"] = float(bool(post_chems & ACIDS) or generic_acid)
    out["polar_solvent_post_present"] = float(bool(post_chems & POLAR_SOLVENTS))
    out["post_identified_chemical_count"] = float(len(post_chems))
    # Explicit concentrations in post-treatment text are separate coordinates.
    for chem in _chemicals(post_text):
        for alias in CHEMICAL_ALIASES[chem]:
            match = re.search(r"(?:" + alias + r")\s*[:=]?\s*([\d.]+\s*(?:vol\.?\s*%|wt\.?\s*%|%))", post_text)
            if match:
                dose, unit = _parse_dose(match.group(1))
                if dose is not None:
                    _put_numeric(out, f"post_{chem}_dose_{unit}", dose)
                    break


def _extra_features(out: dict, row: Mapping, observed: bool) -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for item in row.get("extra_scalar_features") or []:
        if not isinstance(item, Mapping):
            continue
        name = _slug(item.get("name"))
        if FORBIDDEN_EXTRA_PATTERN.search(name):
            continue
        is_observed = name in OBSERVED_EXTRA_NAMES or bool(OBSERVED_EXTRA_PATTERN.search(name))
        if name not in RECIPE_EXTRA_NAMES and not (observed and is_observed):
            continue
        value = _number(item.get("value"))
        prefix = "structure_extra" if is_observed else "extra"
        unit = _text(item.get("unit")) or "unreported"
        key = f"{prefix}_{name}__{_slug(unit) or 'unreported'}"
        out[f"has_{key}"] = float(value is not None)
        out[f"{prefix}_{name}_unit"] = unit
        if value is not None:
            grouped.setdefault((name, unit), []).append(value)
    for (name, unit), values in grouped.items():
        is_observed = name in OBSERVED_EXTRA_NAMES or bool(OBSERVED_EXTRA_PATTERN.search(name))
        key = f"{'structure_extra' if is_observed else 'extra'}_{name}__{_slug(unit)}"
        if len(set(values)) != 1:
            out[f"has_{key}"] = 0.0
            out[f"conflicting_{key}"] = 1.0
            continue
        value = values[0]
        out[key] = value
        out[f"has_{key}"] = 1.0
        aliases = {"coating_stage_temperature": "coating_temperature_C",
            "post_coating_stage_temperature": "post_temperature_C",
            "coating_speed": "coating_speed_m_s", "post_coating_speed": "post_speed_m_s"}
        if name in aliases:
            converted = None
            u = unit.replace("°", "").replace(" ", "")
            if name.endswith("temperature"):
                if u in {"c", "degc", "celsius"}:
                    converted = value
                elif u in {"k", "kelvin"}:
                    converted = value - 273.15
            else:
                factors = {"m/s": 1.0, "mm/s": .001, "cm/s": .01,
                    "mm/min": .001 / 60, "m/min": 1 / 60, "cm/min": .01 / 60}
                if u in factors:
                    converted = value * factors[u]
            if converted is not None:
                _put_numeric(out, aliases[name], converted)


@lru_cache(maxsize=1)
def _molecular_registry() -> dict[str, dict]:
    if not DESCRIPTOR_PATH.exists():
        return {}
    entries = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8")).get("entries", {})
    return {key: item["values"] for key, item in entries.items()
            if isinstance(item, dict) and item.get("status") == "verified" and "values" in item}


def _molecular_features(out: dict) -> None:
    registry = _molecular_registry()
    for stage in ("recipe", "post"):
        molecules = [key for key in registry if out.get(f"{stage}_{key}_present", 0.0) > 0]
        out[f"{stage}_molecule_descriptor_count"] = float(len(molecules))
        out[f"{stage}_unmapped_chemical_count"] = float(sum(
            out.get(f"{stage}_{key}_present", 0.0) > 0 for key in CHEMICAL_ALIASES if key not in registry))
        # Unweighted descriptors of identified PRESENT molecules only. No
        # guessed dose or unknown commercial-product molecular structure.
        for descriptor in ("mol_wt", "mol_logp", "tpsa", "hbd", "hba"):
            values = [_number(registry[key].get(descriptor)) for key in molecules]
            values = [value for value in values if value is not None]
            out[f"has_{stage}_molecule_{descriptor}_mean"] = float(bool(values))
            if values:
                out[f"{stage}_molecule_{descriptor}_mean"] = sum(values) / len(values)
                out[f"{stage}_molecule_{descriptor}_min"] = min(values)
                out[f"{stage}_molecule_{descriptor}_max"] = max(values)


def build_features(row: Mapping[str, Any], *, include_observed: bool = False,
                   include_context: bool = False,
                   chemical_descriptors: bool = True) -> dict[str, float | str]:
    """Build scalar recipe coordinates; never read targets or specimen IDs.

    ``include_observed`` is explicitly post-characterization mode.
    ``include_context`` adds paper/DOI conditioning, unsuitable for claiming
    source-independent generalization. Neither mode includes sample/series IDs.
    """
    out: dict[str, float | str] = {}
    out["polymer_grade"] = _grade(row.get("polymer_grade"))
    for grade in ("ph1000", "ph500", "ai4083", "ai7580", "baytron_p", "icp1020", "sv3"):
        out[f"grade_{grade}"] = float(grade in out["polymer_grade"].split("+"))
    _categories(out, "deposition", row.get("deposition_method"), {
        "spin_coating": r"spin.coat", "blade_coating": r"blade|doctor.blade",
        "drop_casting": r"drop.cast", "casting": r"(?<!drop.)\bcast(?:ing)?\b",
        "screen_printing": r"screen.print", "inkjet_printing": r"ink.?jet",
        "spray_coating": r"spray", "dip_coating": r"dip.coat",
        "wet_spinning": r"wet.spinn", "electrospinning": r"electrospinn",
        "direct_ink_writing": r"direct.ink|\bdiw\b", "slot_die": r"slot.die",
        "electrodeposition": r"electrodepos|electropolymer",})
    _categories(out, "substrate", row.get("substrate"), {
        "glass": r"glass", "silicon": r"silicon|\bsi\b|sio2|wafer",
        "pet": r"\bpet\b|polyethylene.terephthalate", "pen": r"\bpen\b",
        "pdms": r"pdms|polydimethylsiloxane", "pi": r"polyimide|kapton|\bpi\b",
        "ito": r"\bito\b|indium.tin", "fto": r"\bfto\b", "metal": r"gold|silver|steel|copper|aluminum|aluminium",
        "paper": r"paper|cellulose", "textile": r"textile|fabric|cotton",
        "free_standing": r"free.stand|unsupported",})
    _categories(out, "geometry", row.get("specimen_geometry"), {
        "film": r"film|coating", "fiber": r"fibre|fiber|yarn|filament",
        "trace": r"trace|line|printed", "bulk": r"bulk|pellet|block",
        "gel": r"gel|hydrogel", "porous": r"aerogel|foam|porous",
        "vertical_junction": r"vertical|via|out.of.plane",})
    _categories(out, "measurement", row.get("measurement_method"), {
        "four_point": r"four.point|4.point|4pp|van.der.pauw", "two_point": r"two.point|2.point|two.terminal",
        "hall": r"hall", "impedance": r"impedance|\beis\b", "dc": r"\bdc\b|direct.current",
        "in_plane": r"in.plane|lateral", "out_of_plane": r"out.of.plane|vertical|through.plane",})
    out["measurement_out_of_plane"] = float(bool(out["measurement_out_of_plane"] or out["geometry_vertical_junction"]))
    for field in NUMERIC_FIELDS:
        _put_numeric(out, field, row.get(field))
    for stage in ("dry", "anneal", "post_treatment"):
        _put_numeric(out, f"{stage}_duration_s", (_number(row.get(f"{stage}_duration_min")) * 60
            if _number(row.get(f"{stage}_duration_min")) is not None else None))
    _put_numeric(out, "measurement_temperature_K", (out["measurement_temperature_C"] + 273.15
        if "measurement_temperature_C" in out else None))
    for key in ("coating_temperature_C", "coating_speed_m_s", "post_temperature_C", "post_speed_m_s"):
        _put_numeric(out, key, None)
    if "post_treatment_temperature_C" in out:
        _put_numeric(out, "post_temperature_C", out["post_treatment_temperature_C"])
    _ratio_features(out, row.get("pedot_pss_ratio_raw"))
    basis = row.get("dose_basis")
    out["recipe_dose_basis"] = _dose_basis(basis)
    names = row.get("additive_names") or []
    amounts = row.get("additive_amounts_raw") or []
    if not isinstance(names, list):
        names = [names]
    if not isinstance(amounts, list):
        amounts = [amounts]
    out["recipe_amount_alignment_known"] = float(len(names) == len(amounts))
    for chem in CHEMICAL_ALIASES:
        out[f"recipe_{chem}_present"] = 0.0
    unknown_count = 0
    present_count = 0
    for i, name in enumerate(names):
        chemicals = _chemicals(name)
        if _text(name) in {"", "none", "no additive", "pristine", "untreated"}:
            continue
        # Index association is trusted only for equal-length arrays. A single
        # combined chemical name never receives a guessed split dose.
        amount = amounts[i] if len(names) == len(amounts) else None
        dose, unit = _parse_dose(amount, basis)
        present = dose != 0 if dose is not None else True
        present_count += int(present)
        if not chemicals:
            unknown_count += int(present)
        for chem in chemicals:
            out[f"recipe_{chem}_present"] = max(out[f"recipe_{chem}_present"], float(present))
            out[f"recipe_{chem}_dose_unit"] = unit
            _put_numeric(out, f"recipe_{chem}_dose_{unit}", dose if len(chemicals) == 1 else None)
            if chem in {"dmso", "eg"} and unit == "volpct" and len(chemicals) == 1:
                _put_numeric(out, f"{chem}_dose_volpct", dose)
    for chem in ("dmso", "eg"):
        out.setdefault(f"has_{chem}_dose_volpct", 0.0)
    out["recipe_identified_chemical_count"] = sum(out[f"recipe_{chem}_present"] for chem in CHEMICAL_ALIASES)
    out["recipe_unknown_material_count"] = float(unknown_count)
    out["recipe_reported_present_additive_count"] = float(present_count)
    out["recipe_acid_present"] = float(any(out[f"recipe_{chem}_present"] for chem in ACIDS))
    out["recipe_polar_solvent_present"] = float(any(out[f"recipe_{chem}_present"] for chem in POLAR_SOLVENTS))
    _process_features(out, row)
    _extra_features(out, row, include_observed)
    if chemical_descriptors:
        _molecular_features(out)
    if include_observed:
        for field in OBSERVED_FIELDS:
            _put_numeric(out, "structure_" + field, row.get(field))
    if include_context:
        for key, feature_name in (("paper_id", "context_paper"), ("doi", "context_doi")):
            value = row.get(key)
            if value is not None and str(value).strip():
                out[feature_name] = str(value).strip().lower()
    return out
