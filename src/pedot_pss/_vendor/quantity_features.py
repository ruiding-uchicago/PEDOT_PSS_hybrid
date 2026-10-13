"""Measurement-quantity features from a closed, outcome-free input allowlist.

No targets, conductivity/resistivity values, row/sample/series identifiers,
locators, association notes, review flags, or feature-evidence text are read.
An electrical probe assay does not prove that ionic contributions are absent.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata


INPUT_FIELDS = (
    "measurement_method", "specimen_geometry", "polymer_grade",
    "measurement_direction", "measurement_quantity",
)
FEATURE_VERSION = "quantity-input-only-v1"


def _text(value):
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).lower().replace("−", "-").replace("–", "-")


def quantity_features(original_row):
    """Return scalar/categorical features without outcome or identity access.

    ``quantity_class`` is ionic/electronic/mixed/unspecified. For unlabelled
    probe methods, electronic denotes an electrical probe assay; its separate
    basis/explicitness indicators preserve this weaker inference. Missing
    direction or geometry dimensions are never recovered from sample IDs.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("original_row must be a mapping")
    values = {key: _text(original_row.get(key)) for key in INPUT_FIELDS}
    method, geometry = values["measurement_method"], values["specimen_geometry"]
    assay = " ".join((method, geometry, values["measurement_quantity"]))
    ionic = bool(re.search(r"\bionic\b|\bion[- ](?:transfer|transport|conduct)|\bprotonic\b", assay))
    electronic = bool(re.search(r"\belectronic\b|\belectron[- ](?:transport|conduct)", assay))
    mixed = bool(re.search(r"\bmixed[- ](?:ionic|electronic)|\bmixed\s+conduc", assay))
    electrical_probe = bool(re.search(
        r"\b(?:four|4|two|2)[- ](?:point|probe|terminal|line)\b|\bvan\s+der\s+pauw\b|\bhall\b", method))
    if mixed or (ionic and electronic):
        kind, basis = "mixed", "explicit_carrier_labels"
    elif ionic:
        kind, basis = "ionic", "explicit_carrier_label"
    elif electronic:
        kind, basis = "electronic", "explicit_carrier_label"
    elif electrical_probe:
        kind, basis = "electronic", "electrical_probe_assay"
    else:
        kind, basis = "unspecified", "unknown"

    direction_text = " ".join((method, geometry, values["measurement_direction"]))
    vertical = bool(re.search(r"out[- ]of[- ]plane|\bvertical\b|through[- ](?:plane|thickness)|\bperpendicular\b", direction_text))
    inplane = bool(re.search(r"in[- ]plane|\blateral\b|parallel\s+to\s+(?:the\s+)?(?:film|substrate)", direction_text))
    direction = "both" if vertical and inplane else "out_of_plane" if vertical else "in_plane" if inplane else "unspecified"
    material = " ".join((values["polymer_grade"], geometry))
    pedot = bool(re.search(r"pedot|clevios|baytron|\bph\s*(?:1000|500)\b|\bai\s*4083\b", material))
    explicit_control = bool(re.search(r"\b(?:pure|bare)\s+(?:dae|ito|graphene)\b|\bwithout\s+pedot\b", material))
    material_class = "conflicting" if pedot and explicit_control else "pedot_reported" if pedot else "non_pedot_reference" if explicit_control else "unspecified"
    out = {
        "quantity_class": kind,
        "quantity_class_basis": basis,
        "quantity_carrier_explicit": float(ionic or electronic or mixed),
        "quantity_electrical_probe_assay": float(electrical_probe),
        "quantity_ionic": float(kind == "ionic"),
        "quantity_electronic": float(kind == "electronic"),
        "quantity_mixed": float(kind == "mixed"),
        "quantity_unspecified": float(kind == "unspecified"),
        "quantity_direction": direction,
        "quantity_direction_explicit": float(vertical or inplane),
        "quantity_out_of_plane": float(vertical),
        "quantity_in_plane": float(inplane),
        "quantity_material_class": material_class,
        "quantity_pedot_reported": float(pedot),
        "quantity_non_pedot_reference_explicit": float(explicit_control),
        "quantity_nonohmic_protocol_explicit": float(bool(re.search(r"atypical|non[- ]?ohmic|non[- ]?linear", method))),
        "quantity_bias_sweep_protocol": float(bool(re.search(r"bias\s+sweep|voltage\s+sweep", method))),
        "has_quantity_evaluation_bias_V": 0.0,
        "has_quantity_via_diameter_um": 0.0,
    }
    bias = re.search(r"(?:value|evaluated|measured)\s+at\s*([+-]?\d+(?:\.\d+)?)\s*v\b", method)
    if bias:
        out["quantity_evaluation_bias_V"] = float(bias.group(1))
        out["has_quantity_evaluation_bias_V"] = 1.0
    diameter = re.search(r"(?:via\s+diameter|diameter\s+of\s+(?:the\s+)?via)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(nm|um|μm|mm)\b", geometry)
    if diameter:
        value = float(diameter.group(1)) * {"nm": 0.001, "um": 1.0, "μm": 1.0, "mm": 1000.0}[diameter.group(2)]
        out["quantity_via_diameter_um"] = value
        out["has_quantity_via_diameter_um"] = 1.0
    return out
