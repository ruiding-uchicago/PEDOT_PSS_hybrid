"""Pure input-only adapter for the reviewed salt-drop/rinse/redry ledger.

No property defaults, identities, outcomes, models or IO. Caller supplies the
reviewed signed cation-softness registry, not conductivity-derived values.
Only ready, fully bound contact facts are active. Bath molarity is not film
ionic strength, uptake or pH; contact hotplate T is not measured film T.
Unknown redry duration stays unknown and is never used as contact duration.
"""
from collections import Counter
from collections.abc import Mapping
from numbers import Real

import numpy as np

from .post_stage_ledger import SALTS, post_stage_ledger

RAW_FIELDS = ("post_treatment", "process_sequence", "post_treatment_temperature_C",
              "post_treatment_duration_min")


def prepare_salt_protocol_inputs(original_rows, *, cation_softness):
    if not isinstance(original_rows, (list, tuple)) or not original_rows:
        raise ValueError("Nonempty ordered input-row list required")
    if not isinstance(cation_softness, Mapping) or not set(cation_softness) <= set(SALTS):
        raise ValueError("Explicit supported-salt property mapping required")
    registry = {}
    for salt, value in cation_softness.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("Registry values must be finite signed numeric softness")
        registry[salt] = float(value)
    n = len(original_rows)
    coordinates = {key: np.full(n, np.nan) for key in (
        "concentration_mol_L", "cation_softness", "contact_temperature_K", "contact_seconds")}
    active = np.zeros(n, bool)
    sparse, reasons, species_counts = [], Counter(), Counter()
    for index, original in enumerate(original_rows):
        if not isinstance(original, Mapping):
            raise TypeError("Every original input row must be a mapping")
        projected = {key: original.get(key) for key in RAW_FIELDS}
        ledger = post_stage_ledger(projected)
        if ledger["audit"]["ready"] is not True:
            reasons.update(ledger["audit"]["reasons"])
            continue
        contact = ledger["events"][0]
        # Generic post T/time aliases are not stage-specific evidence. Require
        # each contact quantity to be independently bound to a contact clause.
        if any(not set(contact[key]["sources"]) & {"post_treatment", "process_sequence"}
               for key in ("temperature_C", "duration_s")):
            reasons["contact_stage_unbound_raw_alias"] += 1
            continue
        salt = contact["species"]
        if salt not in registry:
            reasons["missing_reviewed_cation_softness"] += 1
            continue
        values = dict(concentration_mol_L=contact["solution_concentration_mol_L"]["value"],
            cation_softness=registry[salt], contact_temperature_K=contact["temperature_C"]["value"]+273.15,
            contact_seconds=contact["duration_s"]["value"])
        if not all(np.isfinite(v) for v in values.values()) or values["contact_temperature_K"] <= 0:
            reasons["invalid_kelvin_conversion"] += 1
            continue
        for key, value in values.items():
            coordinates[key][index] = value
        active[index] = True
        species_counts[salt] += 1
        sparse.append(dict(index=index, species=salt, coordinates=values, events=ledger["events"],
                           provenance=ledger["provenance"]))
    coordinates["active"] = active
    return dict(coordinates=coordinates, rows=sparse, audit=dict(n_rows=n,
        n_admitted=int(active.sum()), n_positive_concentration=int(np.sum(active & (coordinates["concentration_mol_L"]>0))),
        species_counts=dict(sorted(species_counts.items())), inactive_reasons=dict(sorted(reasons.items())),
        no_targets_or_ids_read=True, no_redry_time_imputation=True,
        concentration_basis="applied_solution_not_film_I_pH_or_uptake"))
