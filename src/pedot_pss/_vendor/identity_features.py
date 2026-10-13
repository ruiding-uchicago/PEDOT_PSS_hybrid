"""Bounded additive identity ablation, independent of the frozen recipe builder.

Only original additive names, aligned numeric doses and a closed dose-basis parser
are used. Typography is normalized, but names are not merged across compounds,
salts, products, isomers or ambiguous abbreviations. Family flags describe names,
not mechanisms, measured properties or descriptors.
"""
from __future__ import annotations

import re
import math
import unicodedata
from collections.abc import Mapping

from .features import _dose_basis, _parse_dose

MAX_NAME_LENGTH = 100
FEATURE_VERSION = "identity-additive-names-v1"

# Reject outcome narratives and identifiers even when accidentally placed into
# additive_names. Formula digits, product grades, Ox1 and PEG200 remain legitimate
# identity labels; they are not assumed to be resolved molecular structures.
_POLLUTION = re.compile(
    r"conductiv|resistiv|resistan|\bsigma\b|σ|\bresults?\b|\btargets?\b|"
    r"\bsample\b|sample[_ -]?id|\bspecimen\b|\bseries\b|\breplicate\b|"
    r"row[_ -]?id|paper[_ -]?id|source[_ -]?id|\bdoi\b|\bexperiment\b|"
    r"seebeck|power\s*factor|carrier\s*(?:density|concentration|mobility)|"
    r"sheet\s*resist|\bs\s*/\s*(?:cm|m)\b|ohm|Ω|https?://", re.I)
_IDENTIFIER = re.compile(
    r"^(?:[nbo]\d{3}(?:[-_:].*)?|r\d{2,}|s\d{1,3}|"
    r"(?:row|sample|specimen|series|replicate|experiment|exp|test|condition)"
    r"\s*[#:_-]?\s*[a-z0-9][a-z0-9 _-]*)$", re.I)
_MIXTURE_NARRATIVE = re.compile(r"[=<>≤≥]|\b(?:increased|decreased|reaches|measured|reported|figure|table)\b", re.I)
_EMPTY = {"", "none", "no additive", "not reported", "unknown", "n/a", "pristine", "untreated"}

# Twenty controlled, transparent families. Acronym-only classes retain the word
# 'label' where the name alone cannot establish full chemical identity.
FAMILY_PATTERNS = {
    "trifluoroacetic_acid": r"\btrifluoroacetic\s+acid\b|^tfa$|\bcf3cooh\b",
    "triflic_acid": r"\btriflic\s+acid\b|\btrifluoromethanesul(?:f|ph)onic\s+acid\b|^tfoh$|\bcf3so3h\b",
    "camphorsulfonic_acid": r"\bcamphor(?:sul(?:f|ph)onic|\s+sul(?:f|ph)onic)\s+acid\b",
    "csa_label": r"^(?:[dl][- ])?csa$",
    "sulfonic_acid": r"sul(?:f|ph)onic\s+acid\b|^dbsa$",
    "imidazolium_salt": r"imidazolium\b|\b(?:emim|bmim|bemim)\s*[:/ -]\s*[a-z0-9]|^bmimots$",
    "pyrrolidinium_salt": r"pyrrolidinium\b",
    "choline_or_des_label": r"\bcholine\s+chloride\b|\bchcl\b|\bdes\b",
    "lithium_compound": r"\blithium\b|^li\s+salt$|^li(?:tfsi|beti|clo4|bf4|no3|2so4|b\(|cl\b|br\b)",
    "alkali_hydroxide": r"\b(?:sodium|potassium|lithium)\s+hydroxide\b|^(?:naoh|koh|lioh)$",
    "glycidyl_epoxy_reagent": r"glycid(?:yl|oxy|yloxy)|\bepoxy\b|^bdde$",
    "acrylate_methacrylate": r"acrylate\b|\bpegda\b|\bpegdma\b",
    "oxetane_named_or_series_label": r"oxetane|^ox[1-5]$",
    "photoinitiator_named": r"\bphotoinitiator\b",
    "persulfate": r"\bpersulfate\b|\b(?:na2s2o8|k2s2o8)\b",
    "mxene": r"\bmxene\b|ti3c2t[xz]\b",
    "carbon_nitride": r"\bcarbon\s+nitride\b|\bc3n4\b",
    "metal_nanostructure": r"(?:silver|gold|\bag\b|\bau\b)[ -]*nano(?:wire|particle|rod)|\b(?:agnws?|agnps?|aunps?)\b",
    "surfactant_product_label": r"\b(?:tween\s*\d+|triton|zonyl|dynol|capstone|pluronic)\b|^p123$",
    "polyether_polymer_label": r"\bpoly\s*\(?\s*(?:ethylene\s+(?:oxide|glycol)|ethyleneoxide|ethyleneglycol)|\b(?:peo|peg)(?:\d+)?\b",
}
_FAMILIES = {key: re.compile(pattern, re.I) for key, pattern in FAMILY_PATTERNS.items()}
_SALT_QUALIFIER = re.compile(r"\bsalt\b|\b(?:sodium|potassium|lithium|ammonium)\b", re.I)

# Used ONLY to associate duplicate doses, while exact identity names stay distinct.
# These are explicit compound equivalences, not acronym/fuzzy-name guesses.
_DOSE_ALIASES = {
    "dimethyl_sulfoxide": ("dmso", "dimethyl sulfoxide", "dimethylsulfoxide", "dimethyl sulfoxide (dmso)"),
    "ethylene_glycol": ("eg", "ethylene glycol", "ethylene glycol (eg)"),
    "liclo4": ("liclo4", "lithium perchlorate", "lithium perchlorate (liclo4)"),
    "libf4": ("libf4", "lithium tetrafluoroborate", "lithium tetrafluoroborate (libf4)"),
    "lino3": ("lino3", "lithium nitrate", "lithium nitrate (lino3)"),
    "li2so4": ("li2so4", "lithium sulfate", "lithium sulphate", "lithium sulfate (li2so4)"),
    "licl": ("licl", "lithium chloride", "lithium chloride (licl)"),
    "libr": ("libr", "lithium bromide", "lithium bromide (libr)"),
    "litfsi": ("litfsi", "lithium bis(trifluoromethanesulfonyl)imide (litfsi)"),
    "naoh": ("naoh", "sodium hydroxide", "sodium hydroxide (naoh)"),
    "koh": ("koh", "potassium hydroxide", "potassium hydroxide (koh)"),
}
_DOSE_ENTITY = {alias: entity for entity, aliases in _DOSE_ALIASES.items() for alias in aliases}


def normalize_identity_name(value) -> str | None:
    """Return a bounded, clean whole-name label; no chemical alias substitution."""
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        return None
    name = unicodedata.normalize("NFKC", value).casefold()
    name = name.translate(str.maketrans({"−": "-", "–": "-", "—": "-"}))
    name = re.sub(r"\s+", " ", name).strip()
    if name in _EMPTY or len(name) > MAX_NAME_LENGTH or not re.search(r"[a-z]", name):
        return None
    if _POLLUTION.search(name) or _IDENTIFIER.fullmatch(name) or _MIXTURE_NARRATIVE.search(name):
        return None
    return name


def identity_families(name: str) -> set[str]:
    """Controlled name-family evidence; never equate members' properties."""
    found = {family for family, pattern in _FAMILIES.items() if pattern.search(name)}
    # '...sulfonic acid sodium salt' is not a free acid. Different counterions
    # stay distinct in the exact-name identity and do not enter the acid family.
    if _SALT_QUALIFIER.search(name):
        found.difference_update({"sulfonic_acid", "camphorsulfonic_acid",
                                 "triflic_acid", "trifluoroacetic_acid"})
    return found


def enrich_identity_features(row: Mapping, base_features: Mapping) -> dict:
    """Copy base_features and add additive identity and controlled family presence.

    Only additive names, aligned amounts and the existing closed dose-basis parser
    are used. Amount text never becomes a category. Targets, sample/series IDs,
    process narrative, locators, evidence and review metadata are not read.
    Explicit zero dose has no identity/family presence. Conflicting duplicate
    compound doses remain missing and flagged. Encoding remains fold-local.
    """
    out = dict(base_features)
    names = row.get("additive_names") or []
    if isinstance(names, str):
        names = [names]
    elif not isinstance(names, (list, tuple)):
        names = []
    amounts = row.get("additive_amounts_raw") or []
    if not isinstance(amounts, (list, tuple)):
        amounts = [amounts]
    aligned = len(names) == len(amounts)
    basis_raw = row.get("dose_basis")
    basis = _dose_basis(basis_raw)
    groups = {}
    clean = set()
    for i, value in enumerate(names):
        name = normalize_identity_name(value)
        if name is None:
            continue
        clean.add(name)
        dose, unit = None, "unknown"
        if aligned:
            amount = amounts[i]
            if not isinstance(amount, str) or not _POLLUTION.search(amount):
                dose, unit = _parse_dose(amount, basis_raw)
                if dose is not None and not math.isfinite(dose):
                    dose = None
        entity = _DOSE_ENTITY.get(name, name)
        groups.setdefault(entity, []).append((name, dose, unit))
    found = set()
    present_names = set()
    for entity, items in sorted(groups.items()):
        known = {(unit, dose) for _, dose, unit in items if dose is not None}
        conflict = len(known) > 1
        out["identity_recipe_dose_conflict::" + entity] = float(conflict)
        out["identity_recipe_has_dose::" + entity] = float(len(known) == 1 and not conflict)
        if len(known) == 1 and not conflict:
            unit, dose = next(iter(known))
            out[f"identity_recipe_dose::{entity}::{unit}::{basis}"] = float(dose)
        # Unknown dose is not assumed zero. Multiple inconsistent doses do not
        # create a guessed mean or guessed denominator.
        present = any(dose is None or dose != 0 for _, dose, _ in items)
        for name, _, _ in items:
            out["identity_recipe_name::" + name] = float(present)
            if present:
                present_names.add(name)
                found.update(identity_families(name))
    for family in FAMILY_PATTERNS:
        out["identity_recipe_family_" + family + "_named"] = float(family in found)
    out["identity_recipe_clean_name_count"] = float(len(clean))
    out["identity_recipe_present_name_count"] = float(len(present_names))
    out["identity_recipe_amount_alignment_known"] = float(aligned)
    return out
