"""Closed SWCNT/PEDOT:PSS film inputs; no solver, fit, IO or target access.

x is reported NOMINAL loading mass, never measured final dry/volume fraction.
Filtration can remove PSS, so nominal blend mass must not imply final film mass.
z=0 is direct spin-film formation (precursor nylon filtering is not z=1);
z=1 is actual vacuum-filtered film formation followed by 40MPa/10min cold press.
These are narrowly reviewed protocol shapes, not a general CNT text classifier.
IDs/labels/notes/outputs never decide admission. Locators are metadata only.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
import re
import unicodedata

import numpy as np

VERSION = "closed-SWCNT-nominal-mass-and-film-formation-v1"
INPUT_FIELDS = ("polymer_grade", "specimen_geometry", "additive_names",
    "additive_amounts_raw", "dose_basis", "deposition_method", "process_sequence", "post_treatment")
SCOPE = ("Reported nominal SWCNT mass-loading coordinate, not measured final dry mass, "
    "volume fraction, carrier concentration or conductivity; no source/ID/outcome routing.")
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
FILM = re.compile(r"(?:ultrathin\s+)?(?:PEDOT\s*:\s*PSS/SWCNT|SWCNT/PEDOT\s*:\s*PSS)"
                  r"\s+composite\s+(?:thin\s+)?film\.?", re.I)
GRADE = re.compile(r"(?:Clevios\s+)?PH\s*1000", re.I)
CNT = re.compile(r"SWCNTs?|single[- ]walled carbon nanotubes?", re.I)
OTHER_ADDITIVES = re.compile(r"ethylene glycol|EG|dimethyl sulfoxide|DMSO|"
    r"N,N-dimethylformamide|DMF|Triton X-100", re.I)
ADDITIVE_ROLES = {"ethylene glycol":"eg", "eg":"eg", "dimethyl sulfoxide":"dmso",
    "dmso":"dmso", "n,n-dimethylformamide":"dmf", "dmf":"dmf", "triton x-100":"triton"}
PROTOCOL_DOSE = re.compile(rf"(?P<value>{NUMBER})\s*(?P<unit>wt|vol)\s*%", re.I)
MASS = re.compile(rf"(?P<pct>{NUMBER})\s*wt\s*%(?P<total>\s+of\s+total\s+"
    r"SWCNT\s*\+\s*PEDOT\s*:\s*PSS)?", re.I)
TOTAL_BASIS = re.compile(r"SWCNT wt% is fraction of total SWCNT\s*\+\s*PEDOT:PSS; "
                         r"EG is added to PH1000", re.I)
FILM_BASIS = re.compile(r"Triton X-100 and DMSO in PEDOT:PSS aqueous; "
                        r"SWCNT mass fraction in film", re.I)
FILTER_PRESS = re.compile(rf"(?P<pct>{NUMBER}) wt% SWCNTs dispersed in 50 mL "
    r"(?:dimethyl sulfoxide|N,N-dimethylformamide); "
    r"(?:PH1000 \+ 5 wt% EG homogenized at 15,000 rpm and slowly added|"
    r"same composite preparation as HNTD but high-speed shearing replaced by "
    r"low-velocity magnetic stirring); vacuum filtered, dried, and "
    r"cold pressed at 40 MPa for 10 min\.?", re.I)
SPIN = re.compile(r"Disperse 3 mg SWCNT in 3 ml ethanol; stir at 3000 rpm for 1 h "
    r"and ultrasonicate in ice water for 2 h; filter PEDOT:PSS with 0\.45 um nylon; "
    r"add 0\.03 vol% Triton X-100 and 5 vol% DMSO, stir 1 h; add SWCNT dispersion, "
    r"ultrasonicate 2 h and stir 4 h; spin coat 1000 rpm 30 s then 1500 rpm 30 s; "
    r"dry at 130 C for 5 min\.?", re.I)


def _text(value):
    return unicodedata.normalize("NFKC", value).strip() if isinstance(value, str) and len(value) <= 4000 else ""


def _locators(row):
    values = row.get("locators")
    if not isinstance(values, list): return []
    return [{k: v[k] for k in ("file", "page", "where") if k in v and
             (isinstance(v[k], str) or type(v[k]) is int)}
            for v in values if isinstance(v, Mapping)]


def filler_parallel_input(row):
    """Return one optional mass coordinate and typed forming path; never mutate."""
    if not isinstance(row, Mapping): raise TypeError("Original input row must be a mapping")
    view = {k: _text(row.get(k)) for k in INPUT_FIELDS if k not in
            ("additive_names", "additive_amounts_raw")}
    result = dict(active=False, x_nominal_mass_fraction=None, z_filtration_coldpress=None,
        mass_basis=None, scope=SCOPE, reason="unclosed_composite_film", provenance=[], locators=_locators(row))
    def reject(reason):
        result["reason"] = reason
        return result
    if not FILM.fullmatch(view["specimen_geometry"]) or not GRADE.fullmatch(view["polymer_grade"]):
        return result
    names, amounts = row.get("additive_names"), row.get("additive_amounts_raw")
    if (not isinstance(names, list) or not isinstance(amounts, list) or
            not names or len(names) != len(amounts)):
        return reject("unaligned_additive_amounts")
    names = [_text(n) for n in names]
    index = [i for i, name in enumerate(names) if CNT.fullmatch(name)]
    if len(index) != 1 or any(not CNT.fullmatch(n) and not OTHER_ADDITIVES.fullmatch(n) for n in names):
        return reject("unclosed_material_mixture")
    roles = ["swcnt" if CNT.fullmatch(n) else ADDITIVE_ROLES[n.casefold()] for n in names]
    if len(set(roles)) != len(roles): return reject("duplicate_chemical_entries")
    mass = MASS.fullmatch(_text(amounts[index[0]]))
    if mass is None: return reject("nonpoint_or_untyped_SWCNT_mass")
    pct = float(mass["pct"])
    x = pct/100.
    if not math.isfinite(pct) or not 0. < pct < 100. or not 0. < x < 1.:
        return reject("nonpositive_pure_or_nonfinite_loading")
    post = row.get("post_treatment")
    if post is not None and (not isinstance(post, str) or post.strip()):
        return reject("unreviewed_later_treatment")
    basis, method, process = (view[k] for k in ("dose_basis", "deposition_method", "process_sequence"))
    forming = FILTER_PRESS.fullmatch(process)
    if (mass["total"] and TOTAL_BASIS.fullmatch(basis) and
            method.casefold() == "vacuum filtration and cold pressing" and forming):
        if float(forming["pct"]) != pct: return reject("process_loading_conflict")
        z, mass_basis = 1, "nominal_total_SWCNT_plus_PEDOT_PSS_mass_fraction"
    elif (not mass["total"] and FILM_BASIS.fullmatch(basis) and
            method.casefold() == "spin coating" and SPIN.fullmatch(process)):
        z, mass_basis = 0, "reported_nominal_film_mass_loading"
    else: return reject("unclosed_mass_basis_or_film_formation")
    expected = {"eg":(5., "wt")} if z else {"dmso":(5., "vol"), "triton":(.03, "vol")}
    consistency = {}
    for role, amount in zip(roles, amounts):
        if role == "swcnt": continue
        if role not in expected: return reject("unreviewed_protocol_additive")
        if amount is None or (isinstance(amount, str) and not amount.strip()):
            consistency[role] = dict(status="unknown", reported_amount=amount)
            continue  # The closed process supplies context, not a backfilled amount entry.
        dose = PROTOCOL_DOSE.fullmatch(_text(amount))
        if dose is None: return reject("unclear_protocol_additive_consistency")
        value, unit = float(dose["value"]), dose["unit"].casefold()
        if not math.isfinite(value) or (value, unit) != expected[role]:
            return reject("protocol_additive_dose_conflict")
        consistency[role] = dict(status="explicit_match", reported_amount=_text(amount))
    result.update(active=True, x_nominal_mass_fraction=x, z_filtration_coldpress=z,
        mass_basis=mass_basis, reason="explicit_composite_mass_and_formation",
        provenance=[dict(rule="closed_nominal_mass_and_formation", input_fields=list(INPUT_FIELDS),
            reported_SWCNT_amount=_text(amounts[index[0]]), reported_mass_basis=basis,
            nonCNT_amount_consistency=consistency,
            formation="vacuum_filtered_film_then_cold_pressed" if z else "direct_spin_coated_film",
            precursor_filtration_is_film_formation=False)])
    return result


def filler_parallel_inputs(original_rows, *, row_ids=None):
    """Aligned arrays; unknown x=NaN/z=-1 with active=False, never known zero."""
    rows = [filler_parallel_input(row) for row in original_rows]
    if row_ids is not None:
        row_ids = list(row_ids)
        if (len(row_ids) != len(rows) or any(not isinstance(i, str) or not i for i in row_ids)
                or len(set(row_ids)) != len(row_ids)):
            raise ValueError("Unique metadata row IDs must align with original inputs")
    active = np.asarray([r["active"] for r in rows], dtype=bool)
    return dict(version=VERSION, row_ids=row_ids, active=active,
        x_nominal_mass_fraction=np.asarray([r["x_nominal_mass_fraction"] if r["active"] else np.nan for r in rows]),
        z_filtration_coldpress=np.asarray([r["z_filtration_coldpress"] if r["active"] else -1 for r in rows], dtype=int),
        rows=rows, counts=dict(n_input=len(rows), n_active=int(active.sum()),
            forming_path=dict(Counter(str(r["z_filtration_coldpress"]) for r in rows if r["active"])),
            mass_basis=dict(Counter(r["mass_basis"] for r in rows if r["active"])),
            reasons=dict(Counter(r["reason"] for r in rows))), scope=SCOPE)
