"""Isolated input-role overlay; not a physical model or production profile.

No IDs, targets, electrical values, notes, review flags or raw series are read.
Sample text is passed ONLY to the existing absence-conflict/axis fact parser;
new material-role rules never use it. Unknown roles emit no feature value.
Locators and the one ratio-evidence field bind reviewed input facts, not labels.
No carrier, phase fraction, final ion concentration or metal conductivity is
inferred. The caller owns authentication of independently reviewed protocols.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import re
import unicodedata

from .input_fact_corrections import input_fact_corrections

VERSION = "material-role-input-overlay-v1"
INPUT_FIELDS = (
    "polymer_grade", "specimen_geometry", "substrate", "process_sequence",
    "measurement_method", "measurement_direction", "measurement_quantity",
    "pedot_pss_ratio_raw", "sample_label", "additive_names",
)
AUXILIARY_INPUT_FIELDS = ("feature_evidence.pedot_pss_ratio_raw", "locators.file")
PROTOCOL_FIELDS = ("kind", "quote", "locator.file")
ROLES = ("typed_material_pedot_presence", "typed_material_pssna_role",
         "typed_material_current_path", "typed_measurement_basis")
ABSENCE_KEYS = frozenset(("typed_input_pedot_presence",
    "typed_input_zero_pedot_matrix_identity", "cv_label_material_family",
    "quantity_material_class", "quantity_pedot_reported",
    "quantity_non_pedot_reference_explicit"))
DIRECTION_KEYS = frozenset(("measurement_direction", "measurement_out_of_plane",
    "cv_label_measurement_direction_in_plane", "quantity_direction",
    "quantity_direction_explicit", "quantity_out_of_plane", "quantity_in_plane",
    "typed_input_measurement_direction"))
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PEDOT = r"pedot\s*:\s*pss"
PEDOT_OR_GRADE = r"\b(?:pedot(?:\s*:\s*pss)?|clevios|baytron|ph\s*(?:1000|500))\b"
PSSNA = r"\bpssna\b|poly\(sodium[- ]?4[- ]styrene\s+sulfonate\)"
NEGATION = re.compile(r"\b(?:no|not|without|never)\b", re.I)
FINAL_CAST = re.compile(
    rf"drop[- ]cast\s+(?P<wtpct>{NUMBER})\s*wt\s*%\s+PSSNa\s+in\s+"
    r"(?:deionized|DI)\s+water\s+onto\s+(?:planar\s+)?"
    r"glass(?:\s*/\s*Au)?\s+electrode\s+substrate\.?", re.I)
TEXTILE_BASIS = re.compile(
    r"conductivity calculated from thickness and surface resistance,\s*"
    r"treating the fabric as a thin film with a uniform surface\.?", re.I)


def _text(value):
    if not isinstance(value, str) or len(value) > 8000:
        return ""
    return unicodedata.normalize("NFKC", value).strip()


def _positive_pedot_mention(text, pattern=PEDOT):
    # Only immediately scoped PEDOT negation is negative. "without Au" must
    # never suppress a separate positive PEDOT mention; ambiguous scope vetoes
    # an absence assignment rather than fabricating presence or absence.
    return any(not re.search(r"\b(?:no|not|without|never)\s+(?:(?:a|any)\s+)?$",
                             text[:match.start()], re.I)
               for match in re.finditer(pattern, text, re.I))


def _input_view(row):
    """Restrict the reused parser too; no unrestricted evidence or notes."""
    view = {key: _text(row.get(key)) for key in INPUT_FIELDS if key != "additive_names"}
    names = row.get("additive_names")
    view["additive_names"] = [_text(x) for x in names] if isinstance(names, list) else []
    evidence = row.get("feature_evidence")
    view["feature_evidence"] = {"pedot_pss_ratio_raw": _text(
        evidence.get("pedot_pss_ratio_raw"))} if isinstance(evidence, Mapping) else {}
    locators = row.get("locators")
    view["locators"] = [{"file": _text(x.get("file"))} for x in locators
                         if isinstance(x, Mapping)] if isinstance(locators, list) else []
    return view


def material_role_overlay(original_row, *, protocol_facts=()):
    """Return sparse features/raw_updates plus roles/reasons/provenance.

    Known role strings are bounded vocabulary, not long text categories. Reused
    old-key corrections are explicit replacements for a future matched input
    version, NOT appended aliases that silently change a frozen profile.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("original_row must be a mapping")
    if not isinstance(protocol_facts, (list, tuple)):
        raise TypeError("protocol_facts must be a list or tuple of reviewed facts")
    protocols = []
    for fact in protocol_facts:
        if not isinstance(fact, Mapping):
            raise TypeError("Each reviewed protocol fact must be a mapping")
        locator = fact.get("locator")
        protocols.append({"kind": _text(fact.get("kind")), "quote": _text(fact.get("quote")),
                          "locator": {"file": _text(locator.get("file"))}
                          if isinstance(locator, Mapping) else {}})
    row = _input_view(original_row)
    roles = dict.fromkeys(ROLES, "unknown")
    features, raw_updates, reasons, provenance = {}, {}, [], []

    def record(rule, fields, facts):
        reasons.append(rule)
        provenance.append(dict(rule=rule, input_fields=list(fields), input_facts=facts))

    # Only these two branches of the existing audited parser are admitted.
    old = input_fact_corrections(row, protocol_facts=protocols)
    pure_dae = bool(re.fullmatch(r"pure DAE film\.?", row["specimen_geometry"], re.I))
    positive_geometry = _positive_pedot_mention(row["specimen_geometry"])
    positive_additive = any(re.search(PEDOT, name, re.I) for name in row["additive_names"])
    if (old["features"].get("typed_input_pedot_presence") == "explicit_absent"
            and not pure_dae and not (positive_geometry or positive_additive)):
        features.update({k: v for k, v in old["features"].items() if k in ABSENCE_KEYS})
        roles[ROLES[0]] = "explicit_absent"
        record("existing_bound_pedot_absence", ("process_sequence", "pedot_pss_ratio_raw",
            "feature_evidence.pedot_pss_ratio_raw"), {"absence": True,
            "zero_matrix": features.get("typed_input_zero_pedot_matrix_identity", "unknown")})
    if old["features"].get("typed_input_measurement_direction") == "out_of_plane":
        features.update({k: v for k, v in old["features"].items() if k in DIRECTION_KEYS})
        raw_updates["measurement_direction"] = "out_of_plane"
        record("existing_reviewed_transverse_z", ("sample_label", "measurement_direction",
            "specimen_geometry", "measurement_method", "locators.file"),
            {"axis": "out_of_plane", "source_parser": old["version"],
             "protocol_binding": "existing_input_fact_corrections"})

    grade, geometry, process, method, substrate = (
        row[k] for k in ("polymer_grade", "specimen_geometry", "process_sequence",
                        "measurement_method", "substrate"))
    materials = "; ".join((grade, geometry, process, *row["additive_names"]))
    pssna = bool(re.search(PSSNA, materials, re.I))
    cast = FINAL_CAST.fullmatch(process)
    valid_cast = (cast is not None and math.isfinite(float(cast["wtpct"]))
                  and 0 < float(cast["wtpct"]) <= 100)
    # A PSSNa dispersant/precursor is not evidence that PEDOT is absent.
    product = bool(re.search(rf"(?:^|;\s*){PEDOT}\s+prepared by oxidative "
        r"polymerization(?:\s*;|\s*$)", process, re.I) or
        re.match(rf"PSSNa[- ]based\s+{PEDOT}\s+dispersion\b", process, re.I))
    if pssna and product and not NEGATION.search(process):
        roles[ROLES[1]] = "precursor_or_component_of_pedot_product"
        roles[ROLES[0]] = "explicit_present"
        record("pssna_in_explicit_pedot_product", ("polymer_grade", "process_sequence"),
               {"pssna": "precursor_or_component", "final_product": "pedot_pss"})
    elif (pssna and valid_cast and re.search(PSSNA, grade, re.I)
          and not re.search(PEDOT, materials, re.I)):
        roles[ROLES[1]] = "explicit_final_pssna_specimen"
        record("explicit_pssna_only_cast", ("polymer_grade", "process_sequence", "additive_names"),
               {"cast_species": "pssna", "not_a_carrier_or_fraction_assignment": True})

    active = re.fullmatch(rf"(?:Au|gold)\s*/\s*{PEDOT}\s+double[- ]layer\s+"
                          r"circuit\s+(?:line|trace)\.?", geometry, re.I)
    corroborated = (re.search(r"\b(?:Au|gold)\s+(?:film|line)\b", process, re.I)
        and re.search(rf"(?:covered|coated)\s+with\s+{PEDOT}", process, re.I)
        and re.search(r"\bcracks?\b", process, re.I))
    contact = re.search(r"\b(?:Au|gold)\s+(?:(?:top|bottom|planar)\s+)?electrodes?\b",
                        "; ".join((geometry, substrate, process)), re.I)
    sandwich = re.search(rf"\bAu\s*/\s*{PEDOT}\s*/\s*Au\s+overlap\b", geometry, re.I)
    if active and corroborated and not NEGATION.search(process):
        roles[ROLES[2]] = "active_au_pedot_double_layer"
        roles[ROLES[0]] = "explicit_present"
        record("active_au_pedot_circuit", ("specimen_geometry", "process_sequence"),
               {"path": "cracked_au_plus_pedot", "metal_fraction": "unknown"})
    elif not active and (contact or sandwich) and not NEGATION.search(
            "; ".join((geometry, substrate, process))):
        roles[ROLES[2]] = "au_measurement_contacts"
        record("explicit_au_contacts", ("specimen_geometry", "substrate", "process_sequence"),
               {"role": "contacts_not_parallel_metal_fraction"})
    commercial = (re.match(r"Ag\s*/\s*AgCl\s+(?:disposable|commercial|reference)\s+electrode\b",
                           geometry, re.I)
        and re.fullmatch(r"Commercial reference electrode used in the ECG measurement arrangement\.?",
                         process, re.I))
    if commercial and roles[ROLES[2]] == "unknown" and not re.search(PEDOT, materials, re.I):
        roles[ROLES[2]] = "commercial_ag_agcl_reference"
        record("commercial_ag_agcl_reference", ("specimen_geometry", "process_sequence"),
               {"object": "commercial_reference_not_pedot_recipe"})
    textile = bool(re.search(r"\b(?:textile|fabric|cotton|lycra|polyester)\b", substrate, re.I))
    if textile and TEXTILE_BASIS.fullmatch(method):
        roles[ROLES[3]] = "effective_uniform_film_textile"
        record("explicit_textile_effective_basis", ("substrate", "measurement_method"),
               {"basis": "reported_uniform_film_calculation", "not_intrinsic_dense_film": True})
    if pure_dae:
        conflict = any(_positive_pedot_mention(text, PEDOT_OR_GRADE)
                       for text in (grade, process, *row["additive_names"]))
        if conflict:
            roles[ROLES[0]] = roles[ROLES[2]] = "unknown"
        else:
            roles[ROLES[0]], roles[ROLES[2]] = "explicit_absent", "pure_dae_reference"
            record("explicit_pure_dae_geometry", ("specimen_geometry", "polymer_grade",
                "additive_names", "process_sequence"), {"material": "pure_dae",
                "positive_pedot_conflict": False, "physical_parameters_inferred": False})
    features.update({key: value for key, value in roles.items() if value != "unknown"})
    return dict(version=VERSION, roles=roles, features=features, raw_updates=raw_updates,
        reasons=reasons, provenance=provenance, audit=dict(
            no_carrier_or_volume_fraction_inferred=True, no_targets_or_ids_read=True,
            no_physical_parameters_changed=True, production_connection=False,
            direction_requires_raw_feature_rebuild=bool(raw_updates)))
