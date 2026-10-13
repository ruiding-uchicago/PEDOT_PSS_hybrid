"""Sparse, evidence-bound input corrections; not a production feature profile.

Only closed semantic facts are extracted from input fields. IDs, conductivity,
errors, review flags, distribution labels and outcome evidence are never read.
Sample/series text is used only for explicit measurement state/direction or a
whole concentration statement, never exported as a label. Locator text binds
independently reviewed protocol evidence; it is not a predictive feature.

Raw corrections MUST precede a fresh raw-input feature build. Updating an old
derived dictionary cannot retract solvent/thermal aliases consistently.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math
import re
from typing import Mapping


VERSION = "evidence-bound-input-facts-v1"
PREFIX = "typed_input_"
SCALAR = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
SOLVENT = r"ethanol|methanol|isopropanol|ipa"
BEFORE = re.compile(
    rf"(?:[a-z][a-z -]{{0,70}} )?(?:before|prior to) "
    rf"(?P<solvent>{SOLVENT}) vapou?r (?:treatment|annealing)", re.I)
NEGATIVE_PEDOT = re.compile(r"(?:no|without)\s+PEDOT\s*:\s*PSS", re.I)
ZERO_RATIO = re.compile(
    r"PEDOT:PSS/(?P<matrix>SF|PEO|PVA|PAAm) mass ratio set as 0/100\.", re.I)
CONCENTRATION = re.compile(
    rf"(?P<stock>PH\s*1000|PH\s*500|AI\s*4083) concentration "
    rf"(?P<value>{SCALAR}) wt\s*%", re.I)
SCOPE_CONCENTRATION = re.compile(
    rf"(?:Supplementary Fig\. \d+ x-axis: )?"
    rf"(?P<stock>PH\s*1000|PH\s*500|AI\s*4083) concentration "
    rf"(?P<value>{SCALAR}) wt\s*%; this is formulation concentration, "
    r"not the commercial-stock solids specification\.", re.I)
RAW_FIELDS = frozenset(("post_treatment", "post_treatment_temperature_C",
                        "post_treatment_duration_min", "measurement_direction",
                        "feature_evidence.post_treatment"))
OLD_FEATURE_KEYS = frozenset((
    "cv_label_material_family", "quantity_material_class", "quantity_pedot_reported",
    "quantity_non_pedot_reference_explicit", "measurement_direction",
    "measurement_out_of_plane", "cv_label_measurement_direction_in_plane",
    "quantity_direction", "quantity_direction_explicit", "quantity_out_of_plane",
    "quantity_in_plane"))


def _text(row, key):
    value = row.get(key)
    return value.strip() if isinstance(value, str) and len(value) <= 8000 else ""


def _evidence(row, key):
    evidence = row.get("feature_evidence")
    return _text(evidence, key) if isinstance(evidence, Mapping) else ""


def _locators(row):
    locators = row.get("locators")
    return deepcopy([dict(x) for x in locators if isinstance(x, Mapping)]) \
        if isinstance(locators, list) else []


def _stock(text):
    return re.sub(r"\s+", "", text).casefold()


def _provenance(row, rule, fields, *, protocol=None):
    record = {"rule": rule, "input_evidence": fields, "locators": _locators(row)}
    if protocol is not None:
        record["reviewed_protocol_fact"] = deepcopy(protocol)
    return record


def input_fact_corrections(original_row, *, protocol_facts=()):
    """Return raw_updates/features/provenance/audit, without mutating the row.

    ``protocol_facts`` is an explicit independently reviewed evidence input.
    A transverse-axis fact needs kind='transverse_axis_definition', the exact
    input-only quote 'transverse (z-direction)', and a locator.file matching a
    row locator. This does not prove an arbitrary supplied quote is authentic;
    callers must review/bind that small evidence artifact before using it.
    Ambiguous stage/axis/material/basis omits corrections, rather than guessing.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("original_row must be a mapping")
    row = original_row
    raw, features, provenance, statuses = {}, {}, [], {}
    label, series = _text(row, "sample_label"), _text(row, "series_label")
    process, post = _text(row, "process_sequence"), _text(row, "post_treatment")

    # The whole post-treatment must BEGIN with the named later vapor event.
    # Never remove an earlier anneal/crosslinking scalar by matching its value.
    before = BEFORE.fullmatch(series)
    if before is not None:
        solvent = before["solvent"].casefold()
        start = re.match(rf"{re.escape(solvent)}\s+vapou?r\s+(?:annealing|treatment)\b", post, re.I)
        contradicted = re.search(r"\bafter\b", label, re.I)
        sequence_has_vapor = re.search(r"\bvapou?r\b", process, re.I)
        checked_process = _evidence(row, "process_sequence")
        evidence_post = _evidence(row, "post_treatment")
        extras = row.get("extra_scalar_features") or []
        unbound_alias = not isinstance(extras, list)
        unbound_alias |= any(isinstance(x, Mapping) and re.search(
            r"post|vapou?r", _text(x, "name"), re.I) for x in extras)
        unbound_alias |= any(row.get(key) is not None for key in (
            "post_temperature_C", "post_duration_s", "post_treatment_duration_s"))
        unbound_alias |= any(str(key).startswith("post_") and key not in RAW_FIELDS
                             and row.get(key) is not None for key in row)
        if (start and process and checked_process == process and not sequence_has_vapor
                and not contradicted and not unbound_alias
                and (not evidence_post or evidence_post == post)):
            raw.update(post_treatment=None, post_treatment_temperature_C=None,
                       post_treatment_duration_min=None)
            if evidence_post:
                raw["feature_evidence.post_treatment"] = None
            features[PREFIX + "measurement_before_vapor_treatment"] = 1.0
            features[PREFIX + "later_vapor_solvent_identity"] = solvent
            statuses["before_vapor"] = "later_post_event_retracted_prior_anneal_preserved"
            provenance.append(_provenance(row, "explicit_before_named_vapor_event", {
                "series_label": series, "process_sequence": process,
                "feature_evidence.process_sequence": checked_process,
                "post_treatment": post}))
        else:
            statuses["before_vapor"] = "ambiguous_or_unbound_post_event_unchanged"

    # Absence requires a whole negated process clause, not a substring anywhere.
    clauses = [x.strip().rstrip(".") for x in process.split(";")]
    negative = [x for x in clauses if NEGATIVE_PEDOT.fullmatch(x)]
    ratio_text = _evidence(row, "pedot_pss_ratio_raw")
    ratio = ZERO_RATIO.fullmatch(ratio_text)
    raw_ratio = _text(row, "pedot_pss_ratio_raw")
    zero_ratio = ratio is not None and re.fullmatch(r"0\s*/\s*100", raw_ratio) is not None
    positive_clauses = [x for x in clauses if re.search(r"PEDOT\s*:\s*PSS", x, re.I)
                        and not NEGATIVE_PEDOT.fullmatch(x)]
    grade = _text(row, "polymer_grade")
    positive_grade = re.search(r"pedot|clevios|baytron|\bph\s*(?:1000|500)\b", grade, re.I)
    positive_label = (re.search(r"PEDOT\s*:\s*PSS", label, re.I)
                      and not NEGATIVE_PEDOT.fullmatch(label))
    ratio_conflict = bool(raw_ratio and not zero_ratio)
    if negative or zero_ratio:
        if not positive_clauses and not positive_grade and not positive_label and not ratio_conflict:
            features.update({PREFIX + "pedot_presence": "explicit_absent",
                             "cv_label_material_family": "non_pedot_control",
                             "quantity_material_class": "non_pedot_reference",
                             "quantity_pedot_reported": 0.0,
                             "quantity_non_pedot_reference_explicit": 1.0})
            if zero_ratio:
                features[PREFIX + "zero_pedot_matrix_identity"] = ratio["matrix"].casefold()
            statuses["pedot_absence"] = "explicit_absence_not_carrier_assignment"
            provenance.append(_provenance(row, "explicit_negation_or_bound_zero_endmember", {
                "process_sequence": process, "pedot_pss_ratio_raw": raw_ratio,
                "feature_evidence.pedot_pss_ratio_raw": ratio_text}))
        else:
            statuses["pedot_absence"] = "conflicting_presence_or_unbound_ratio_unknown"

    direction = _text(row, "measurement_direction")
    direction_text = "; ".join((label, direction, _text(row, "specimen_geometry"),
                                _text(row, "measurement_method")))
    transverse = bool(re.search(r"\btransverse\b", direction_text, re.I))
    direct_z = bool(re.fullmatch(r"transverse\s*\(z-direction\)", direction, re.I))
    row_files = {x.get("file") for x in _locators(row) if isinstance(x.get("file"), str)}
    bound_facts, conflicting_definition = [], False
    for fact in protocol_facts:
        if not isinstance(fact, Mapping) or fact.get("kind") != "transverse_axis_definition":
            continue
        locator = fact.get("locator")
        if isinstance(locator, Mapping) and locator.get("file") in row_files:
            if (isinstance(fact.get("quote"), str) and re.fullmatch(
                    r"transverse\s*\(z-direction\)", fact["quote"].strip(), re.I)):
                bound_facts.append(fact)
            else:
                conflicting_definition = True
    in_plane = re.search(r"in[- _]plane|\blateral\b", direction_text, re.I)
    if transverse:
        if (direct_z or bound_facts) and not in_plane and not conflicting_definition:
            raw["measurement_direction"] = "out_of_plane"
            features.update({"measurement_direction": "out_of_plane",
                "measurement_out_of_plane": 1.0,
                "cv_label_measurement_direction_in_plane": 0.0,
                "quantity_direction": "out_of_plane", "quantity_direction_explicit": 1.0,
                "quantity_out_of_plane": 1.0, "quantity_in_plane": 0.0,
                PREFIX + "measurement_direction": "out_of_plane"})
            statuses["transverse_direction"] = "reviewed_z_definition_bound"
            provenance.append(_provenance(row, "transverse_with_explicit_z_definition", {
                "sample_label": label, "measurement_direction": direction,
                "specimen_geometry": _text(row, "specimen_geometry"),
                "measurement_method": _text(row, "measurement_method")},
                protocol=bound_facts[0] if bound_facts else None))
        else:
            statuses["transverse_direction"] = "unbound_or_conflicting_axis_unknown"

    # A stated formulation wt% is NOT stock solids or measured dry-film vf.
    concentration = CONCENTRATION.fullmatch(label)
    scope_text = _evidence(row, "stock_solids_scope")
    scope = SCOPE_CONCENTRATION.fullmatch(scope_text)
    if concentration is not None:
        valid = False
        if scope is not None:
            try:
                value = Decimal(concentration["value"])
                supported = Decimal(scope["value"])
                numeric = float(value)
                valid = (value == supported and 0 <= value <= 100 and math.isfinite(numeric)
                         and not (value > 0 and numeric == 0)
                         and _stock(concentration["stock"]) == _stock(scope["stock"]))
                known_grade = re.findall(r"PH\s*1000|PH\s*500|AI\s*4083", grade, re.I)
                valid = valid and all(_stock(x) == _stock(concentration["stock"])
                                      for x in known_grade)
            except (InvalidOperation, OverflowError):
                valid = False
        if valid:
            features.update({PREFIX + "formulation_stock_dispersion_wt_percent": numeric,
                PREFIX + "formulation_stock_dispersion_identity": _stock(concentration["stock"]),
                PREFIX + "formulation_concentration_basis": "reported_formulation_wt_percent"})
            statuses["formulation_concentration"] = "explicit_formulation_not_stock_solids"
            provenance.append(_provenance(row, "bound_formulation_stock_dispersion_wt_percent", {
                "sample_label": label, "feature_evidence.stock_solids_scope": scope_text}))
        else:
            statuses["formulation_concentration"] = "basis_or_value_unbound_unknown"

    return {"version": VERSION, "raw_updates": raw, "features": features,
            "provenance": provenance, "audit": {"rules": statuses,
                "raw_feature_rebuild_required": bool(raw),
                "prior_anneal_fields_never_removed": True,
                "no_targets_or_outcome_evidence_read": True,
                "no_carrier_or_dry_volume_fraction_inferred": True}}


def rebuild_corrected_features(original_row, feature_builder, *, protocol_facts=()):
    """Apply raw patch to a copy, freshly build raw features, then apply facts.

    The caller's builder must rebuild ALL raw-input features for its profile,
    not update a cached dictionary; it must NOT produce physical/domain derived
    coordinates. Compute those only from this function's returned features.
    This module deliberately does not choose or import a production profile.
    """
    result = input_fact_corrections(original_row, protocol_facts=protocol_facts)
    if not set(result["raw_updates"]).issubset(RAW_FIELDS):
        raise ValueError("Unregistered raw correction")
    corrected = deepcopy(dict(original_row))
    for key, value in result["raw_updates"].items():
        if key == "feature_evidence.post_treatment":
            corrected["feature_evidence"] = dict(corrected.get("feature_evidence") or {})
            corrected["feature_evidence"]["post_treatment"] = value
        else:
            corrected[key] = value
    built = feature_builder(corrected)
    if not isinstance(built, Mapping):
        raise TypeError("feature_builder must return a mapping")
    if any(str(key).startswith(("coord_", "domain_", "computed_")) for key in built):
        raise ValueError("Builder already produced derived physics; rebuild raw features first")
    if any(not (key.startswith(PREFIX) or key in OLD_FEATURE_KEYS) for key in result["features"]):
        raise ValueError("Unregistered feature correction")
    if any(key.startswith(PREFIX) for key in built):
        raise ValueError("Builder unexpectedly contains an existing correction overlay")
    return {"original_row": corrected, "features": {**built, **result["features"]},
            "corrections": result}


