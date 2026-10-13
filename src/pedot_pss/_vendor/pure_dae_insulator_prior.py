"""UV-on pure-DAE PRIOR sensitivity, not identified photoconductivity.

The nominal -12 log10(S/cm) comes from the old effective-low-phase numerical
prior, NOT a measured DAE endmember or a target fit. Neither temperature nor
light intensity/photogeneration is modeled. Dark/UV-off is outside this scope.
This posthoc development expert replaces the complete five-field prediction;
it does not retain an inappropriate PEDOT residual or a diluted physics alpha.
No IO, fitting, IDs, labels, electrical values or evidence text are read.
"""
from collections.abc import Mapping
import math
import numbers
import re
import unicodedata

import numpy as np

from .material_role_overlay import material_role_overlay

VERSION = "pure-dae-uv-on-insulator-prior-v1"
NOMINAL_LOG10_SIGMA_S_CM = -12.0
SENSITIVITY_LOG10_SIGMA_S_CM = (-16.0, -14.0, -12.0, -10.0, -8.0, -6.0)
INPUT_FIELDS = ("polymer_grade", "specimen_geometry", "additive_names",
                "process_sequence", "measurement_method")
SCOPE = ("POSTHOC SAME-OOF DEVELOPMENT prior sensitivity; only input-explicit "
         "UV-on pure DAE, not identified DAE conductivity or new validation. "
         "Quantitative material/temperature/light-intensity response unidentified.")
_UV_ON = re.compile(r"\buv(?:\s+(?:light|state))?\s+on\b")
_OFF_DARK = re.compile(r"\buv(?:\s+(?:light|state))?\s+off\b|\bdark\b")
_NEG_UV = re.compile(
    r"\b(?:no|not|without|never)\s+(?:(?:using|under|with)\s+)?"
    r"(?:\d+(?:\.\d+)?\s*nm\s+)?uv\b|"
    r"\buv(?:\s+(?:light|state))?\s+(?:(?:is|was)\s+)?(?:not|never)\b")
_NEG_AFTER_ON = re.compile(r"^\s+(?:(?:is|was)\s+)?(?:not|never)\s+(?:used|applied|present)\b")


def _text(value):
    if not isinstance(value, str) or len(value) > 8000:
        return ""
    return re.sub(r"[-\u2010-\u2015\u2212]", " ",
                  unicodedata.normalize("NFKC", value)).lower().strip()


def pure_dae_admission(original_row):
    """Closed material-role + explicit UV-on admission; no predictive IDs.

    Only five raw fields are projected to the existing broader role parser.
    Missing material/illumination facts are inactive, never imputed.
    """
    if not isinstance(original_row, Mapping):
        raise TypeError("original_row must be a mapping")
    row = {key: original_row.get(key) for key in INPUT_FIELDS}
    result = dict(active=False, reason="material_not_closed_pure_dae",
                  illumination_basis="unknown", version=VERSION, scope=SCOPE)
    names = row["additive_names"]
    if ((names is not None and (not isinstance(names, list) or
            any(not isinstance(name, str) for name in names))) or
            any(value is not None and not isinstance(value, str)
                for key, value in row.items() if key != "additive_names")):
        return dict(result, reason="malformed_input")
    roles = material_role_overlay(row)["roles"]
    if (roles["typed_material_current_path"] != "pure_dae_reference" or
            roles["typed_material_pedot_presence"] != "explicit_absent"):
        return result
    texts = [_text(row[key]) for key in ("process_sequence", "measurement_method")]
    if any(_OFF_DARK.search(text) or _NEG_UV.search(text) or
           any(_NEG_AFTER_ON.search(text[match.end():]) for match in _UV_ON.finditer(text))
           for text in texts):
        return dict(result, reason="off_dark_or_negated_uv_conflict")
    fields = [key for key, text in zip(("process_sequence", "measurement_method"), texts)
              if _UV_ON.search(text)]
    if not fields:
        return dict(result, reason="uv_on_not_explicit")
    return dict(result, active=True, reason="closed_pure_dae_explicit_uv_on",
                illumination_basis="+".join(fields))


def compose(parent, admitted, polybot_mask, *, log10_sigma_prior=NOMINAL_LOG10_SIGMA_S_CM):
    """Replace admitted rows by [prior, prior, prior, 0, 0], alpha=1.

    Fields: prediction, raw physics, weighted physics, residual, weighted domain.
    Sensitivity arms use one fixed scalar each; no selection/fit is performed.
    Outside rows and all PB rows are copied exactly; admitting PB is an error.
    """
    if (not isinstance(log10_sigma_prior, numbers.Real) or
            isinstance(log10_sigma_prior, (bool, np.bool_)) or
            not math.isfinite(float(log10_sigma_prior)) or
            float(log10_sigma_prior) not in SENSITIVITY_LOG10_SIGMA_S_CM):
        raise ValueError("Choose one declared finite scalar prior sensitivity")
    values, mask, pb = np.asarray(parent), np.asarray(admitted), np.asarray(polybot_mask)
    if (values.ndim != 2 or values.shape[1] != 5 or values.dtype.kind != "f" or
            not np.isfinite(values).all() or mask.shape != (len(values),) or
            pb.shape != mask.shape or mask.dtype != bool or pb.dtype != bool):
        raise ValueError("Finite floating five-field parent and aligned Boolean masks required")
    if np.any(mask & pb):
        raise ValueError("Pure-DAE prior cannot admit a Polybot row")
    if np.any(np.abs(values[:, 0] - values[:, 2] - values[:, 3]) > 1e-10):
        raise ValueError("Parent weighted-physics/residual decomposition differs")
    result = values.copy()
    result[mask] = (log10_sigma_prior, log10_sigma_prior, log10_sigma_prior, 0., 0.)
    return result
