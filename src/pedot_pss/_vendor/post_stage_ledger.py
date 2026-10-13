"""Input-only, narrow salt-drop/contact -> rinse -> redry event ledger.

Solution molarity is NOT film ionic strength, pH or uptake. No ID/label/result,
evidence narrative or model is read. Unknown/conflicting facts remain explicit.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import math
import re
import unicodedata

VERSION = "explicit-post-salt-stage-ledger-v1"
SALTS = ("CuCl2", "AgNO3", "InCl3", "LiCl", "NaCl", "MgCl2", "NiCl2")
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
START = r"(?<![\w.^×*+\-])"
MARKER = re.compile(r"\b(?P<contact>drop(?:ped)?(?:[ -]cast)?)\b|\b(?P<rinse>rins(?:e|ed|ing))\b|\b(?P<redry>redr(?:y|ied)|dry\s+again)\b", re.I)
FORMULA = re.compile(r"\b(?:" + "|".join(SALTS) + r")\b", re.I)
OUTCOME = re.compile(r"conductiv|resist|seebeck|\bsigma\b|σ|power.?factor", re.I)
DEPOSITION = re.compile(r"spin[ -]?coat|blade[ -]?coat|\bdepos(?:it|ited|ition)\b", re.I)
ROLES = ("contact", "rinse", "redry")


def _text(value):
    text = unicodedata.normalize("NFKC", value).replace("−", "-").replace("µ", "u").replace("μ", "u") if isinstance(value, str) else ""
    return re.sub(r"°\s*C\b", "C", text)


def _number(value, *, temperature=False):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        exact = Decimal(str(value)); number = float(exact)
    except (InvalidOperation, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or (number == 0 and exact != 0):
        return None
    return number if (number > -273.15 if temperature else number >= 0) else None


def _field(candidates=(), *, bad=None):
    known = [item for item in candidates if item[0] is not None]
    if bad:
        return {"status": bad}
    if not known:
        return {"status": "unknown"}
    value = known[0][0]
    if any(not math.isclose(value, v, rel_tol=1e-12, abs_tol=0.) for v, _, _ in known):
        return {"status": "conflict"}
    return dict(status="known", value=value, approximate=any(a for _, a, _ in known),
                sources=[s for _, _, s in known])


def _quantity(text, pattern, factors, source, *, temperature=False):
    values = []
    range_pattern = START + rf"{NUM}\s*(?:[-–—]|(?i:to|or))\s*{NUM}\s*(?:{pattern})(?![\w/^])"
    bad = "alternative_or_bound" if re.search(range_pattern, text) else None
    for match in re.finditer(START + rf"(?P<n>{NUM})\s*(?P<u>{pattern})(?![\w/^])", text):
        before, after = text[max(0, match.start()-22):match.start()], text[match.end():match.end()+22]
        if (re.search(r"[<>≤≥]=?\s*$|\d\s*(?:[-–—]|to|or)\s*$|\b(?:not|no|without)\s*$", before, re.I)
                or re.match(r"\s*(?:[-–—]|to|or)\s*[+\-]?\d", after, re.I)):
            bad = "alternative_or_bound"; continue
        number = _number(match["n"], temperature=temperature)
        value = None if number is None else number * factors[match["u"]]
        if value is None or not math.isfinite(value) or (value == 0 and number != 0):
            bad = "invalid"; continue
        approx = bool(re.search(r"(?:about|approximately|approx\.?|[~≈])\s*$", before, re.I))
        values.append((value, approx, source))
    return _field(values, bad=bad)


def _merge(fields, explicit=None):
    bad = next((f["status"] for f in fields if f["status"] not in ("known", "unknown")), None)
    values = [(f["value"], f["approximate"], s) for f in fields if f["status"] == "known" for s in f["sources"]]
    if explicit is not None:
        value, source, temperature = explicit
        number = _number(value, temperature=temperature)
        if value is not None:
            if number is None: bad = "invalid"
            else: values.append((number, False, source))
    return _field(values, bad=bad)


def _water(text):
    return bool(re.search(r"\baqueous\b|\bwater\b", text, re.I) and not re.search(
        r"\bnon[ -]?aqueous\b|\b(?:without|no|not)\s+(?:(?:a|an|any|with|DI|deionized)\s+)*(?:aqueous|water)\b|\bwater[ -]?free\b|\bor\b|ethanol|methanol|\b(?:DMSO|DMF|THF|IPA)\b|glycol|acetone", text, re.I))


def _events(text, source):
    matches = list(MARKER.finditer(text)); events = []
    for i, match in enumerate(matches):
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        clause = text[match.start():end]; role = match.lastgroup
        if role == "contact" and not re.search(r"\b(?:" + "|".join(SALTS) + r"|salt)\s+solution\b", clause, re.I):
            continue
        before = text[max(0, match.start()-35):match.start()]
        stage_bad = bool(DEPOSITION.search(clause) or re.search(r"\b(?:immers|soak|wash)\w*\b", clause, re.I))
        stage_bad = stage_bad or (role == "contact" and re.search(r"before\s+(?:film\s+)?(?:coating|deposition)", clause, re.I))
        prior = text[:match.start()]
        if role == "contact" and source == "process_sequence":
            deposits = list(DEPOSITION.finditer(prior))
            stage_bad = stage_bad or not deposits or any(re.search(r"\b(?:no|not|without)\s*$", prior[max(0,d.start()-20):d.start()], re.I) for d in deposits)
        if stage_bad or OUTCOME.search(clause) or re.search(r"\b(?:no|not|without|before)\s+(?:\w+\s+){0,3}$", before, re.I):
            events.append(dict(role=role, invalid=True, source=source)); continue
        events.append(dict(role=role, text=clause, source=source,
                           span=[match.start(), end], invalid=False))
    return events


def post_stage_ledger(original_row):
    """Return events/scalars/audit/provenance; ready is a strict input predicate.

    Only post_treatment/process_sequence and explicit post T/time fields are
    accessed. Input evidence is left to external provenance review, not mined.
    Provenance spans address normalized text, not raw byte positions.
    A unique same-order event chain may join these two descriptions; repeated
    contacts, conflicting aliases, negative/bounded/ranged quantities fail shut.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("Input row must be a mapping")
    if any(original_row.get(k) is not None and not isinstance(original_row.get(k), str) for k in ("post_treatment", "process_sequence")):
        return dict(version=VERSION, events=[], scalars={}, audit=dict(ready=False, reasons=["malformed_text"]), provenance=[])
    sources = {name: _text(original_row.get(name)) for name in ("post_treatment", "process_sequence")}
    chains = {name: _events(text, name) for name, text in sources.items()}
    chains = {name: events for name, events in chains.items() if events}
    reasons = []
    if not chains or not any(any(e["role"] == "contact" for e in es) for es in chains.values()):
        return dict(version=VERSION, events=[], scalars={}, audit=dict(ready=False, reasons=["no_explicit_salt_drop"]), provenance=[])
    for name, events in chains.items():
        if [e["role"] for e in events] != list(ROLES): reasons.append("incomplete_or_repeated_order:"+name)
        if any(e["invalid"] for e in events): reasons.append("negated_outcome_or_stage_event:"+name)
    if reasons:
        return dict(version=VERSION, events=[], scalars={}, audit=dict(ready=False, reasons=reasons), provenance=[])
    by_role = {role: [e for es in chains.values() for e in es if e["role"] == role] for role in ROLES}
    contact = by_role["contact"]; texts = [e["text"] for e in contact]
    formulae = {s for s in SALTS if any(re.search(r"\b"+s+r"\b", t, re.I) for t in texts)}
    species = next(iter(formulae)) if len(formulae) == 1 else None
    if species is None: reasons.append("unknown_or_alternative_salt")
    if not all(_water(t) for t in texts): reasons.append("carrier_unresolved")
    def quantity(events, pattern, factors, **kwargs):
        return _merge([_quantity(e["text"], pattern, factors, e["source"], **kwargs) for e in events])
    concentration = quantity(contact, r"mM|M|mol/L", {"mM": .001, "M": 1., "mol/L": 1.})
    volume = quantity(contact, r"uL|mL", {"uL": 1., "mL": 1000.})
    temperature = _merge([_quantity(e["text"], r"°\s*C|C", {"C": 1., "°C": 1., "° C": 1.}, e["source"], temperature=True) for e in contact],
                         (original_row.get("post_treatment_temperature_C"), "post_treatment_temperature_C", True))
    duration = quantity(contact, r"min|minutes?|s|seconds?", {"min":60., "minute":60., "minutes":60., "s":1., "second":1., "seconds":1.})
    raw_minutes = original_row.get("post_treatment_duration_min")
    if raw_minutes is not None:
        number = _number(raw_minutes)
        duration = _merge([duration], (None if number is None else number*60., "post_treatment_duration_min", False)) if number is not None else {"status":"invalid"}
    rinse = by_role["rinse"]; redry = by_role["redry"]
    water = all(_water(e["text"]) for e in rinse)
    counts = []
    for e in rinse:
        words = re.sub(r"\b(one|two|three|four|five)\b", lambda m: str(("one","two","three","four","five").index(m[1].lower())+1), e["text"], flags=re.I)
        counts.append(_quantity(words, r"times|cycles|×|x", {"times":1., "cycles":1., "×":1., "x":1.}, e["source"]))
    count = _merge(counts)
    if count["status"] == "known" and (count["value"] < 1 or not count["value"].is_integer()): count = {"status":"invalid"}
    dry_temp = quantity(redry, r"°\s*C|C", {"C":1., "°C":1., "° C":1.}, temperature=True)
    dry_time = quantity(redry, r"min|minutes?|s|seconds?", {"min":60.,"minute":60.,"minutes":60.,"s":1.,"second":1.,"seconds":1.})
    fields = dict(solution_concentration_mol_L=concentration, applied_volume_uL=volume, temperature_C=temperature, duration_s=duration)
    events = [dict(role="contact", operation="salt_drop", species=species, carrier="water" if "carrier_unresolved" not in reasons else None, **fields),
              dict(role="rinse", solvent="water" if water else None, repeat_count=count),
              dict(role="redry", temperature_C=dry_temp, duration_s=dry_time)]
    for name, field in fields.items():
        if field["status"] != "known": reasons.append("contact_"+name+":"+field["status"])
    if not water: reasons.append("rinse_solvent_unresolved")
    if count["status"] != "known": reasons.append("rinse_count:"+count["status"])
    if dry_temp["status"] != "known": reasons.append("redry_temperature:"+dry_temp["status"])
    if dry_time["status"] not in ("known", "unknown"): reasons.append("redry_duration:"+dry_time["status"])
    scalars = {}
    for event in events:
        for key, value in event.items():
            if isinstance(value, dict) and value.get("status") == "known":
                scalars["post_stage_"+event["role"]+"_"+key] = value["value"]
                if value["approximate"]: scalars["post_stage_"+event["role"]+"_"+key+"_approximate"] = 1.
    if species is not None and concentration["status"] == "known":
        scalars["post_stage_contact_salt_present"] = float(concentration["value"] > 0)
        scalars["post_stage_contact_salt_species"] = species
    provenance = [{k:e[k] for k in ("role","source","span")} for es in chains.values() for e in es]
    return dict(version=VERSION, events=events, scalars=scalars, audit=dict(ready=not reasons, reasons=reasons,
                concentration_basis="applied_solution_not_film_I_pH_or_uptake"), provenance=provenance)
