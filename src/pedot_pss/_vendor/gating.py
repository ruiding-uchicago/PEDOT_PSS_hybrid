"""Dose-aware gating of the open representation (v0.2.1 fixes from the 1016 literature sanity audit).

Applied by openrep.transform_dict when the spec carries a `gate` config; identical in training and prediction.

  * Trace doses (audit F07): every additive (A) and post-treatment (B) descriptor key is multiplied by a component
    gate in [0, 1]; a component with no entity writes the same keys as 0 (not missing), so a vanishing dose
    approaches "no additive" continuously. Additive gate per entity: 1 - exp(-d / d0) (d0 = 0.05 vol% / wt%,
    0.005 mol/L, 0.5 mg/mL; unknown dose -> 1). Practical doses (>= ~0.2 vol%) keep gate ~1.
  * Acid strength (F01): the gate of an acid in the active post-treatment is a Hill function of its concentration
    (K = 55 wt% or vol%, 8 mol/L; n = 4; unknown concentration -> 1). Treatments at >= 100 C (hot-drop: the acid
    concentrates as water evaporates) use at least the strength of a 90 wt% acid. The GEM acid flag
    `acid_post_present` becomes this strength. Non-acid post chemicals use the same trace gate.
  * Units (F26): DMSO / EG doses given in wt%, mol/L or mmol/L are converted to the GEM's vol% keys
    (vol% = wt% x 1.0 / rho, or c[mol/L] x MW / (10 rho)).
  * Stage slots (F16): position-indexed `planned_stage_NN_*` keys are dropped (the operations they describe stay in
    the order-free process flags and scalars).
  * Repeat counts (F29): `*_cycles` counts are capped (cfg["cycles_cap"]).
  * Unit unification (F26, cfg["unify_units"]): additive doses given as wt% (mixture basis), mol/L, mmol/L or
    mg/mL are converted to vol% before the descriptor aggregation (density of DMSO/EG, else 1.0 g/mL; molar mass
    from the RDKit descriptor), so the learners see the same dose whatever the unit.
  * Thermal jitter (F03/F14, round 2, cfg["jitter"]): see jitter_rows; used by openrep_dropout for the augmented copy.

v2 (round 4, after the re-audit of the round-3 candidate; enabled by cfg["v2"]):
  * N01 hot acid: no floor; a smooth evaporative-concentration factor 1 + 9 sigmoid((T - 120) / 10) multiplies a
    parsed acid concentration (capped at the neat acid), so 99 -> 100 C has no step and a pH 3 bath stays weak.
  * N02 acid concentration from text when the dose parser found none: pH x -> 10^-x mol/L, ppm, g/L, "dilute(d)"
    -> 1 mol/L; no information -> full strength as before (the training corpus' unstated acids are concentrated).
  * N03/N04 an acid mention without its own dose takes the strength of the dosed acid(s) in the same treatment.
  * N05/N12/N13 every additive dose is unified to vol% where possible: solids-basis doses rescaled by the stock
    solids content (stock_solids_wt_percent, default 1.2 wt%), absolute uL/mL/mg via stock_volume, ratios,
    ppm/ppb, "trace" (0.01 vol%), bare numbers as %; dispersion/solution wording is never a solids basis.
  * N09 additives whose trace gate is below 0.5 leave the aggregation and the additive count.
  * F02/N10/N11 under a strong acid the additive signal (descriptors, doses, GEM DMSO/EG doses) is damped by
    1 - lambda * acid_strength (cfg["acid_damps_additive"], 0.7): concentrated acid removes PSS and largely
    overrides secondary doping (training data: additive + acid is ~1.2x acid alone).
  * N07 process flags after deposition (acid, immerse, rinse, anneal) are also derived from the text/scalar
    fields, the description-format flags has_process_sequence / process_stage_boundary_known are dropped, and the
    post-coating stage temperature is filled from post_treatment_temperature_C, so omitting the stage list does
    not change the prediction when the text carries the same steps.
  * N06/N15/N26 temperature inputs are clipped to the training support (post-treatment <= 200 C, others <= 230 C;
    cfg["temp_clip"]): beyond it the model neither rewards nor extrapolates (degradation is not modelled).

v3 (v0.2.2; enabled by cfg["v3"], on top of v2):
  * A post-treatment liquid is described by its composition: each acid by its mass fraction (wt%/vol% as given,
    mol/L x molar mass, pH or words via acid_text_conc; hot-drop evaporation raises it smoothly, capped at 98%), and
    the remainder by the stated co-solvent(s) or, if none is named, water. A 1 wt% aqueous acid is then 99% water
    and is read like a water dip; a concentrated acid stays acid-dominated. Trace non-acid chemicals are dropped.
    The post-treatment descriptors are no longer scaled by a component gate (the composition carries the dilution).
  * Repeat-count features (`*_cycles`) are dropped entirely: a count written in the text does not change the result.

v3b (v0.2.2; cfg["v3b"], on top of v3). The v3 retrain still read a 1 wt% acid bath as an acid treatment, through
features that in the training data only occur with acids (an explicit water dose and a second slot, a water rinse,
and acid flags of ~1e-7 that trees split from 0). v3b makes a weak acid exactly absent:
  * acid strengths below ACID_FLOOR (~21 wt% H2SO4 at room temperature) become 0, above it they rise smoothly
    ((s - floor) / (1 - floor)); this strength weights the acid in the composition and sets acid_post_present and
    the acid operation flags;
  * every bath gets an explicit composition: named solvents without a dose share what the dosed components leave
    (a water dip is 100 vol% water, like the water in a dilute acid bath), else water takes the remainder;
  * dose and saturation keys scale with the component's weight, slots list non-acids before acids, slot unit strings
    are not written, and all post-treatment keys are zero-filled, so a zero-weight acid equals no acid;
  * a water rinse writes no rinse features (it only marks that an acid was used).

Vendored from the m-open study: package-relative imports.
"""
from __future__ import annotations

import math
import re

VERSION = "gate-v3b-1019"
TRACE_D0 = {"volpct": 0.05, "wtpct": 0.05, "molpct": 0.05, "pct_unresolved": 0.05, "pct_basis_unresolved": 0.05,
            "mol_L": 0.005, "mmol_L": 5.0, "normality_eq_L": 0.005, "mg_mL": 0.5}
POST_D0 = dict(TRACE_D0)
ACID_K = {"wtpct": 55.0, "volpct": 55.0, "pct_unresolved": 55.0, "pct_basis_unresolved": 55.0, "molpct": 55.0,
          "mol_L": 8.0, "mmol_L": 8000.0, "normality_eq_L": 16.0}
ACID_N = 4.0
HOT_C = 100.0
HOT_FLOOR = 90.0 ** ACID_N / (ACID_K["wtpct"] ** ACID_N + 90.0 ** ACID_N)
VOLPCT_PER_MOL_L = {"cid:679": 78.13 / (10 * 1.100), "cid:174": 62.07 / (10 * 1.113)}   # DMSO, EG
DENSITY = {"cid:679": 1.100, "cid:174": 1.113}
GEM_DOSE = {"cid:679": "dmso_dose_volpct", "cid:174": "eg_dose_volpct"}
SOLIDS_BASIS = re.compile(r"solid|dry (?:mass|weight)|(?:polymer|pedot(?::pss)?) (?:mass|weight)|"
                          r"relative to (?:the )?(?:pedot|polymer|solid)|per (?:mass|weight) of (?:the )?(?:pedot|polymer)")
_NOT_SOLIDS = re.compile(r"dispersion|solution|stock|ink|suspension|mixture")
STOCK_SOLIDS_DEFAULT = 1.2
NEAT = {"wtpct": 98.0, "volpct": 100.0, "pct_unresolved": 98.0, "pct_basis_unresolved": 98.0, "mol_L": 18.4,
        "mmol_L": 18400.0, "normality_eq_L": 36.8}
_TRACE = re.compile(r"\btrace|\ba drop\b|\bdrops?\b|small amount|minute amount")
_PH = re.compile(r"\bph\s*(?:=|of|at|~|≈)?\s*(\d+(?:\.\d+)?)")
_PPM = re.compile(r"(\d+(?:\.\d+)?)\s*(ppm|ppb)\b")
_GL = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g\s*/\s*l|g\s*l-1|mg\s*/\s*ml)\b")
_DILUTE = re.compile(r"\bdilute(?:d)?\b")
_OP_TEXT = {"immerse": re.compile(r"immers|soak|\bdip|bath|submerg|treated (?:in|with)"),
            "rinse": re.compile(r"rins|wash")}
_FORMAT_KEYS = ("has_process_sequence", "process_stage_boundary_known")
_STAGE_SLOT = re.compile(r"^(?:has_)?planned_stage_\d+_")
_CYCLES = re.compile(r"cycles(?:_|$)")
_KEEP_UNGATED = re.compile(r"(?:_dose|_sat|_unit|sat_sum|n_dose_known|dose_sum_\w+)$")
ACID_FLOOR = 0.02


def soft_acid(s):
    """v3b acid strength: 0 below ACID_FLOOR, then a linear rise to 1 (continuous)."""
    return None if s is None else max(0.0, (float(s) - ACID_FLOOR) / (1.0 - ACID_FLOOR))


def trace_gate(dose, unit, d0_table=TRACE_D0):
    if dose is None or not math.isfinite(dose):
        return 1.0
    if dose <= 0:
        return 0.0
    d0 = d0_table.get(unit)
    return 1.0 if d0 is None else 1.0 - math.exp(-dose / d0)


def acid_strength(dose, unit, hot):
    if dose is None or not math.isfinite(dose):
        s = 1.0
    elif dose <= 0:
        s = 0.0
    else:
        k = ACID_K.get(unit)
        s = 1.0 if k is None else dose ** ACID_N / (k ** ACID_N + dose ** ACID_N)
    return max(s, HOT_FLOOR) if hot and s > 0 else s


def solids_basis(basis):
    b = str(basis or "").lower()
    return bool(SOLIDS_BASIS.search(b)) and not _NOT_SOLIDS.search(b)


def evaporation_factor(row):
    t = row.get("post_treatment_temperature_C")
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 + 9.0 / (1.0 + math.exp(-(t - 120.0) / 10.0))


def acid_strength_v2(dose, unit, factor):
    if dose is None or not math.isfinite(dose):
        return None
    if dose <= 0:
        return 0.0
    k = ACID_K.get(unit)
    if k is None:
        return None
    c = min(dose * factor, NEAT.get(unit, dose * factor))
    return c ** ACID_N / (k ** ACID_N + c ** ACID_N)


def acid_text_conc(text):
    """(dose, unit) of an acid treatment from words when no explicit dose was parsed, else None."""
    t = str(text or "").lower()
    m = _PH.search(t)
    if m:
        return 10.0 ** -float(m.group(1)), "mol_L"
    m = _PPM.search(t)
    if m:
        return float(m.group(1)) / (1e4 if m.group(2) == "ppm" else 1e7), "wtpct"
    m = _GL.search(t)
    if m:
        return float(m.group(1)) / 10.0, "wtpct"
    if _DILUTE.search(t):
        return 1.0, "mol_L"
    return None


def _stock_volume_mL(row):
    for item in row.get("extra_scalar_features") or []:
        if str(item.get("name")) == "stock_volume" and item.get("value") is not None:
            u = str(item.get("unit") or "ml").lower().replace(" ", "")
            f = {"ml": 1.0, "ul": 1e-3, "µl": 1e-3, "μl": 1e-3, "l": 1000.0}.get(u)
            if f:
                return float(item["value"]) * f
    return None


def _raw_amounts(row, chem):
    """CID group -> raw amount text for the additives of a row (aligned lists only)."""
    names = row.get("additive_names") or []
    amounts = row.get("additive_amounts_raw") or []
    if not isinstance(names, list) or not isinstance(amounts, list) or len(names) != len(amounts):
        return {}
    out = {}
    for name, amount in zip(names, amounts):
        if isinstance(name, str):
            out.setdefault(chem.entity(chem.names.get(name), name)["group"], str(amount or "").lower())
    return out


def to_volpct(e, raw, row):
    """vol% of the dispersion for one additive entity, or None when it cannot be converted."""
    d, u, g = e["dose"], e["unit"], e["ent"]["group"]
    rho = DENSITY.get(g, 1.0)
    if d is None or not math.isfinite(d):
        if raw and _TRACE.search(raw):
            return 0.01
        m = _PPM.search(raw or "")
        if m:
            return float(m.group(1)) / (1e4 if m.group(2) == "ppm" else 1e7) / rho
        return None
    if d <= 0:
        return 0.0
    if u == "volpct":
        return d
    if u in ("pct_unresolved", "pct_basis_unresolved", "amount_unresolved"):
        return d
    if u == "wtpct":
        if solids_basis(row.get("dose_basis")):
            solids = row.get("stock_solids_wt_percent") or STOCK_SOLIDS_DEFAULT
            return d * float(solids) / 100.0 / rho
        return d / rho
    if u in ("mol_L", "mmol_L"):
        mw = (e["ent"].get("desc") or {}).get("mol_wt")
        return None if mw is None else (d / 1000.0 if u == "mmol_L" else d) * 10.0 ** mw / (10.0 * rho)
    if u == "mg_mL":
        return d / (10.0 * rho)
    if u == "reported_ratio":
        return 100.0 * d / (1.0 + d)
    v = _stock_volume_mL(row)
    if v:
        if u == "uL":
            return d / (10.0 * v)
        if u == "mL":
            return 100.0 * d / v
        if u == "mg":
            return d / (10.0 * v * rho)
    return None


def unify_units_v2(ents, row, chem):
    raw = _raw_amounts(row, chem)
    out = []
    for e in ents:
        vol = to_volpct(e, raw.get(e["ent"]["group"]), row)
        out.append(dict(e, dose=vol, unit="volpct") if vol is not None and vol <= 100.0 else e)
    return out


def _hot(row):
    t = row.get("post_treatment_temperature_C")
    try:
        return t is not None and float(t) >= HOT_C
    except (TypeError, ValueError):
        return False


def unify_units(ents, row):
    """Copies of additive entities with convertible doses expressed in vol% (mixture basis only)."""
    basis = str(row.get("dose_basis") or "").lower()
    if SOLIDS_BASIS.search(basis):
        return ents
    out = []
    for e in ents:
        d, u, g = e["dose"], e["unit"], e["ent"]["group"]
        rho = DENSITY.get(g, 1.0)
        vol = None
        if d is not None and d > 0:
            if u == "wtpct":
                vol = d / rho
            elif u in ("mol_L", "mmol_L"):
                mw = (e["ent"].get("desc") or {}).get("mol_wt")      # stored as log10(MW)
                if mw is not None:
                    vol = (d / 1000.0 if u == "mmol_L" else d) * 10.0 ** mw / (10.0 * rho)
            elif u == "mg_mL":
                vol = d / (10.0 * rho)
        out.append(dict(e, dose=vol, unit="volpct") if vol is not None and vol <= 100.0 else e)
    return out


def gate_info(row, chem, unify=False, v2=False):
    """Per-row gate quantities from the canonical experiment (helper keys, removed by apply())."""
    if v2:
        return gate_info_v2(row, chem)
    out = {}
    ents = chem.additive_entities(row)
    gated = unify_units(ents, row) if unify else ents
    out["__gate_A"] = max((trace_gate(e["dose"], e["unit"]) for e in gated), default=0.0)
    basis = str(row.get("dose_basis") or "").lower()
    solids_basis = bool(SOLIDS_BASIS.search(basis))   # % of polymer mass: not a mixture %
    for e in ents:
        g = e["ent"]["group"]
        if g in GEM_DOSE and e["dose"] is not None and e["dose"] > 0:
            d, u = e["dose"], e["unit"]
            vol = None
            if u == "wtpct" and not solids_basis:
                vol = d / DENSITY[g]
            elif u in ("mol_L", "mmol_L"):
                vol = (d / 1000.0 if u == "mmol_L" else d) * VOLPCT_PER_MOL_L[g]
            elif u == "mg_mL":
                vol = d / (10.0 * DENSITY[g])
            if vol is not None and vol <= 100.0:
                out["__volpct_" + GEM_DOSE[g]] = vol
    hot = _hot(row)
    active = [e for e in chem.post_entities(row) if not e.get("rinse")]
    gates, acids = [], []
    for e in active:
        if e["ent"]["cls"].get("acid"):
            s = acid_strength(e["dose"], e["unit"], hot)
            acids.append(s)
            gates.append(s)
        else:
            gates.append(trace_gate(e["dose"], e["unit"], POST_D0))
    out["__gate_B"] = max(gates, default=0.0)
    if acids:
        out["__acid_strength"] = max(acids)
    return out


def additive_items_v2(row, chem):
    """Unified, trace-filtered additive entities for the A aggregation (v2)."""
    ents = unify_units_v2(chem.additive_entities(row), row, chem)
    return [e for e in ents if trace_gate(e["dose"], e["unit"]) >= 0.5]


def gate_info_v2(row, chem):
    out = {}
    ents = unify_units_v2(chem.additive_entities(row), row, chem)
    gates = [trace_gate(e["dose"], e["unit"]) for e in ents]
    out["__gate_A"] = max(gates, default=0.0)
    out["__n_A"] = float(sum(g >= 0.5 for g in gates))
    for e in ents:
        g = e["ent"]["group"]
        if g in GEM_DOSE and e["unit"] == "volpct" and e["dose"] is not None and 0 < e["dose"] <= 100.0:
            out["__volpct_" + GEM_DOSE[g]] = e["dose"]
    factor = evaporation_factor(row)
    active = [e for e in chem.post_entities(row) if not e.get("rinse")]
    acids = [e for e in active if e["ent"]["cls"].get("acid")]
    known = [s for s in (acid_strength_v2(e["dose"], e["unit"], factor) for e in acids) if s is not None]
    if known:
        s_unknown = max(known)
    else:
        tc = acid_text_conc(row.get("post_treatment"))
        s_unknown = acid_strength_v2(tc[0], tc[1], factor) if tc else 1.0
    gates_b = []
    for e in active:
        if e["ent"]["cls"].get("acid"):
            s = acid_strength_v2(e["dose"], e["unit"], factor)
            gates_b.append(s_unknown if s is None else s)
        else:
            gates_b.append(trace_gate(e["dose"], e["unit"], POST_D0))
    out["__gate_B"] = max(gates_b, default=0.0)
    if acids:
        out["__acid_strength"] = max([s_unknown if (s := acid_strength_v2(e["dose"], e["unit"], factor)) is None
                                      else s for e in acids])
    text = str(row.get("post_treatment") or "").lower()
    if acids and out["__acid_strength"] > 0.05:
        out["__op_acid"] = 1.0
    for name, pat in _OP_TEXT.items():
        if text and pat.search(text):
            out["__op_" + name] = 1.0
    if row.get("anneal_temperature_C") is not None:
        out["__op_anneal"] = 1.0
    if text and row.get("post_treatment_temperature_C") is not None:
        out["__post_T"] = float(row["post_treatment_temperature_C"])
    return out


def _acid_fraction(e, factor):
    """Mass fraction (0-1) of an acid in its treatment liquid (hot-drop factor applied), or None if unknown."""
    d, u = e.get("dose"), e.get("unit")
    if d is None or not math.isfinite(d):
        return None
    if d <= 0:
        return 0.0
    if u in ("wtpct", "volpct", "pct_unresolved", "pct_basis_unresolved", "molpct"):
        f = d / 100.0
    elif u in ("mol_L", "mmol_L", "normality_eq_L"):
        c = d / 1000.0 if u == "mmol_L" else d / 2.0 if u == "normality_eq_L" else d
        mw = 10.0 ** ((e["ent"].get("desc") or {}).get("mol_wt", 2.0))      # descriptor stores log10(MW)
        f = c * mw / 1000.0                                                 # g per mL, density ~1
    else:
        return None
    return min(0.98, f * factor)


def aggregate_weighted(items, weights, p, v3b=False):
    """chemrep.aggregate with per-entity weights (v3): identity-type keys (class flags, slot descriptors, counts)
    scale with the entity's effective weight, means use saturation x weight; dose/saturation keys stay as given.
    max_/min_ descriptor keys are not written (they would expose a weak component at full value).
    v3b: dose/saturation bookkeeping also scales with the weight, slots list non-acids first, no slot unit strings."""
    from . import chemrep
    out = {p + "n": float(sum(weights))}
    if not items:
        return out
    out[p + "n_struct"] = float(sum(w for e, w in zip(items, weights) if e["ent"]["struct"]))
    sats = []
    for e in items:
        sv = chemrep.saturation(e["dose"], e["unit"])
        sats.append(0.5 if sv is None else sv)
    eff = [sv * w for sv, w in zip(sats, weights)]
    unit_sum = {}
    for e, w in zip(items, weights):
        if e["dose"] is not None and e["unit"] in chemrep.UNIT_GROUPS:
            u = chemrep.UNIT_GROUPS[e["unit"]]
            d = e["dose"] / 1000.0 if e["unit"] == "mmol_L" else e["dose"]
            unit_sum[u] = unit_sum.get(u, 0.0) + (d * w if v3b else d)
    if v3b:
        out[p + "n_dose_known"] = float(sum(w for e, w in zip(items, weights)
                                            if chemrep.saturation(e["dose"], e["unit"]) is not None))
    else:
        out[p + "n_dose_known"] = float(sum(chemrep.saturation(e["dose"], e["unit"]) is not None for e in items))
    out[p + "sat_sum"] = float(sum(eff))
    for u, v in unit_sum.items():
        out[p + "dose_sum_" + u] = v
    for c in chemrep.CLASSES:
        flags = [e["ent"]["cls"].get(c, 0.0) for e in items]
        if any(flags):
            out[p + "cls_" + c] = float(min(1.0, sum(f * w for f, w in zip(flags, weights))))
            out[p + "clsdose_" + c] = float(sum(f * x for f, x in zip(flags, eff)))
    struct = [(e, x) for e, x in zip(items, eff) if e["ent"]["desc"]]
    if struct:
        for k in sorted({k for e, _ in struct for k in e["ent"]["desc"]}):
            vals = [(e["ent"]["desc"][k], x) for e, x in struct if k in e["ent"]["desc"]]
            tot = sum(x for _, x in vals)
            if vals and tot > 1e-12:
                out[p + "wm_" + k] = float(sum(v * x for v, x in vals) / tot)
    polys = [(e, x) for e, x in zip(items, eff) if e["ent"].get("polydesc")]
    if polys:
        for k in ("mol_logp", "tpsa", "hbd", "hba", "fg_ether", "fg_hydroxyl", "fg_sulfonic_acid_or_sulfonate",
                  "fg_amide", "fg_amine", "fg_ester", "positive_charge_count", "negative_charge_count", "log10_mn"):
            vals = [(e["ent"]["polydesc"][k], x) for e, x in polys if k in e["ent"]["polydesc"]]
            tot = sum(x for _, x in vals)
            if vals and tot > 1e-12:
                out[p + "poly_wm_" + k] = float(sum(v * x for v, x in vals) / tot)
    if v3b:
        order = sorted(range(len(items)), key=lambda i: (bool(items[i]["ent"]["cls"].get("acid")), -eff[i],
                                                         items[i]["ent"]["group"]))
    else:
        order = sorted(range(len(items)), key=lambda i: (-eff[i], items[i]["ent"]["group"]))
    for slot, i in enumerate(order[:2], start=1):
        e, w = items[i], weights[i]
        q = f"{p}s{slot}_"
        out[q + "sat"] = float(sats[i] * w if v3b else sats[i])
        if e["dose"] is not None:
            out[q + "dose"] = float(e["dose"] * w if v3b else e["dose"])
        if not v3b:
            out[q + "unit"] = str(e["unit"])
        d = e["ent"]["desc"]
        if d:
            for k in chemrep.SLOT_DESC:
                if k in d:
                    out[q + k] = d[k] * w
        pd = e["ent"].get("polydesc")
        if pd:
            for k in ("mol_logp", "tpsa", "fg_ether", "fg_hydroxyl", "fg_sulfonic_acid_or_sulfonate", "log10_mn"):
                if k in pd:
                    out[q + "poly_" + k] = pd[k] * w
        for c in ("acid", "polar_solvent", "ionic_liquid", "metal_salt", "surfactant", "polymer", "nano_carbon",
                  "nano_metal"):
            if e["ent"]["cls"].get(c):
                out[q + "cls_" + c] = float(w)
    return out


def post_features_v3(row, chem, cfg=None):
    """openB_ features of the post-treatment by composition (v3; role-split like chemrep.post_features).

    Each acid: mass fraction (hot-drop factor applied) as its dose and its Hill strength as its weight; the remainder
    of the liquid: the named co-solvent(s) or water, weight 1; other chemicals: their trace gate as weight.
    v3b (cfg["v3b"]): acid weights through soft_acid, an explicit composition for every bath, no water rinse.
    """
    v3b = bool(cfg and cfg.get("v3b"))
    soft = soft_acid if v3b else (lambda s: s)
    water = chem.entity(chem.post["water"], "water") if chem.post.get("water") is not None else None
    items = chem.post_entities(row)
    rinse = [e for e in items if e.get("rinse")
             and not (v3b and water is not None and e["ent"]["group"] == water["group"])]
    active = [e for e in items if not e.get("rinse")
              and (e["ent"]["cls"].get("acid") or trace_gate(e["dose"], e["unit"], POST_D0) >= 0.5)]
    acids = [e for e in active if e["ent"]["cls"].get("acid")]
    others = [e for e in active if not e["ent"]["cls"].get("acid")]
    factor = evaporation_factor(row)
    fracs = [_acid_fraction(e, factor) for e in acids]
    strengths = [acid_strength_v2(e["dose"], e["unit"], factor) for e in acids]
    known_f = [f for f in fracs if f is not None]
    known_s = [x for x in strengths if x is not None]
    if acids and not known_f:
        tc = acid_text_conc(row.get("post_treatment"))
        if tc:
            known_f = [_acid_fraction({"dose": tc[0], "unit": tc[1], "ent": acids[0]["ent"]}, factor)]
            known_s = [acid_strength_v2(tc[0], tc[1], factor)]
    s_fill = max(known_s) if known_s else 1.0
    comp, weights = [], []
    if acids and known_f:
        f_acid = max(known_f)
        for e, f, st in zip(acids, fracs, strengths):
            comp.append(dict(e, dose=100.0 * (f if f is not None else f_acid), unit="volpct"))
            weights.append(soft(st if st is not None else s_fill))
        dosed = [e for e in others if e["dose"] is not None]
        solvents = [e for e in others if e["dose"] is None]
        for e in dosed:
            comp.append(e)
            weights.append(trace_gate(e["dose"], e["unit"], POST_D0))
        rest = max(0.0, 1.0 - f_acid)
        if solvents:
            for e in solvents:
                comp.append(dict(e, dose=100.0 * rest / len(solvents), unit="volpct"))
                weights.append(1.0)
        elif rest > 0 and water is not None:
            comp.append({"ent": water, "dose": 100.0 * rest, "unit": "volpct", "rinse": False})
            weights.append(1.0)
    elif v3b:
        for e in acids:                                   # acid of unknown concentration: as reported
            comp.append(e)
            weights.append(soft(s_fill))
        dosed = [e for e in others if e["dose"] is not None]
        solvents = [e for e in others if e["dose"] is None]
        for e in dosed:
            comp.append(e)
            weights.append(trace_gate(e["dose"], e["unit"], POST_D0))
        rest = max(0.0, 1.0 - sum(min(1.0, e["dose"] / 100.0) for e in dosed if e["unit"] in ("wtpct", "volpct")))
        if solvents:                                      # a neat solvent dip is 100 vol% of that solvent
            for e in solvents:
                comp.append(dict(e, dose=100.0 * rest / len(solvents), unit="volpct"))
                weights.append(1.0)
        elif dosed and rest > 0 and water is not None:    # a dosed solute without a named solvent: aqueous
            comp.append({"ent": water, "dose": 100.0 * rest, "unit": "volpct", "rinse": False})
            weights.append(1.0)
    else:
        for e in active:
            comp.append(e)
            if e["ent"]["cls"].get("acid"):
                st = acid_strength_v2(e["dose"], e["unit"], factor)
                weights.append(st if st is not None else s_fill)
            else:
                weights.append(trace_gate(e["dose"], e["unit"], POST_D0))
    out = aggregate_weighted(comp, weights, "openB_", v3b=v3b)
    out["openB_rinse_n"] = float(len(rinse))
    for c in ("polar_solvent", "acid", "base", "metal_salt"):
        if any(e["ent"]["cls"].get(c) for e in rinse):
            out["openB_rinse_cls_" + c] = 1.0
    vals = [e["ent"]["desc"] for e in rinse if e["ent"]["desc"]]
    for k in ("mol_wt", "mol_logp", "tpsa", "hbd"):
        v = [d[k] for d in vals if k in d]
        if v:
            out["openB_rinse_mean_" + k] = float(sum(v) / len(v))
    out["__n_B"] = float(len(comp))
    return out


def apply(out, keyspace, cfg):
    """Gate a transformed feature dict in place; `keyspace` = numeric openA_/openB_ keys to fill with 0."""
    ga = out.pop("__gate_A", None)
    gb = out.pop("__gate_B", None)
    acid = out.pop("__acid_strength", None)
    if cfg.get("v3b"):
        acid = soft_acid(acid)
    n_a = out.pop("__n_A", None)
    n_b = out.pop("__n_B", None)
    ops = {k[len("__op_"):]: out.pop(k) for k in [k for k in out if k.startswith("__op_")]}
    post_t = out.pop("__post_T", None)
    v2 = bool(cfg.get("v2"))
    damp = 1.0 - float(cfg.get("acid_damps_additive", 0.0)) * float(acid or 0.0) if v2 else 1.0
    conv = {k[len("__volpct_"):]: out.pop(k) for k in [k for k in out if k.startswith("__volpct_")]}
    if ga is not None:
        for k in keyspace["A"]:
            out.setdefault(k, 0.0)
        for k in [k for k in out if k.startswith("openA_")]:
            v = out[k]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if not _KEEP_UNGATED.search(k):
                    out[k] = float(v) * ga * damp
                elif v2:
                    out[k] = float(v) * damp
        if "identity_recipe_present_name_count" in out:
            out["identity_recipe_present_name_count"] = (n_a * damp if v2 and n_a is not None else
                                                         float(out["identity_recipe_present_name_count"]) * ga)
    if gb is not None:
        mult = (1.0 if (n_b or 0.0) > 0 else 0.0) if cfg.get("v3") else gb
        for k in keyspace["B_all" if cfg.get("v3b") else "B"]:
            out.setdefault(k, 0.0)
        for k in [k for k in out if k.startswith("openB_") and not k.startswith("openB_rinse_")]:
            v = out[k]
            if isinstance(v, (int, float)) and not _KEEP_UNGATED.search(k):
                out[k] = float(v) * mult
    if acid is not None and out.get("acid_post_present"):
        out["acid_post_present"] = float(acid)
    for key, vol in conv.items():
        if not out.get(key):
            out[key] = float(vol)
            out["has_" + key] = 1.0
        pb = "pb_" + key.split("_dose_")[0] + "_volpct_reported"       # Polybot-protocol twin key, if present
        if not out.get(pb) and any(k.startswith("pb_") for k in out):   # families that carry Polybot keys
            out[pb] = float(vol)
            out["pb_has_" + key.split("_dose_")[0] + "_volpct_reported"] = 1.0
    if v2:
        for key in ("dmso_dose_volpct", "eg_dose_volpct", "pb_dmso_volpct_reported", "pb_eg_volpct_reported"):
            if out.get(key):
                out[key] = float(out[key]) * damp
        for name, flag in ops.items():
            out["after_deposition_op_" + name] = max(float(out.get("after_deposition_op_" + name) or 0.0), flag)
        for key in _FORMAT_KEYS:
            out.pop(key, None)
        if post_t is not None and out.get("extra_post_coating_stage_temperature__c") is None:
            out["extra_post_coating_stage_temperature__c"] = post_t
            out["extra_post_coating_stage_temperature_unit"] = "c"
            out["has_extra_post_coating_stage_temperature__c"] = 1.0
        clip = cfg.get("temp_clip") or {"post": 200.0, "other": 230.0}
        for k in [k for k in out if "temperature" in k.lower()]:
            v = out[k]
            if k.startswith("has_") or "unit" in k.lower() or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            cap = clip["post"] if "post" in k.lower() else clip["other"]
            if v > cap:
                out[k] = float(cap)
    if cfg.get("drop_stage_slots", True):
        for k in [k for k in out if _STAGE_SLOT.match(k)]:
            del out[k]
    if cfg.get("v3"):
        for k in [k for k in out if _CYCLES.search(k)]:
            del out[k]
        if acid is not None:                       # acid operation flags follow the acid's strength
            for key in ("post_op_acid", "after_deposition_op_acid"):
                if out.get(key):
                    out[key] = float(min(float(out[key]), acid))
    cap = cfg.get("cycles_cap", 2.0)
    for k in [k for k in out if _CYCLES.search(k) and not k.startswith("has_")]:
        v = out[k]
        if isinstance(v, (int, float)) and v > cap:
            out[k] = float(cap)
    return out


def keyspace(chem, slot_desc, classes, v3=False):
    """All numeric descriptor keys the open representation can write for A and B (fixed, from the table).

    v3 adds the keys of acid + water compositions (slot 2) and of aggregate_weighted; "B_all" (used by v3b) also
    holds the dose/saturation bookkeeping keys. Without v3 the key space is the v0.2.1 one."""
    from . import chemrep
    keys = {"A": set(), "B": set(), "B_all": set()}
    names = list(chem.names.items())
    posts = list(chem.post.items())
    for prefix, comp, recs in (("openA_", "A", names), ("openB_", "B", posts)):
        for name, rec in recs:
            ent = chem.entity(rec, name)
            item = {"ent": ent, "dose": 1.0, "unit": "volpct"}
            pairs = [[dict(item)]]
            if v3 and comp == "B" and chem.post.get("water") is not None:  # acid + water compositions use slot 2
                pairs.append([dict(item), {"ent": chem.entity(chem.post["water"], "water"), "dose": 1.0,
                                           "unit": "volpct"}])
            for group in pairs:
                agg = chemrep.aggregate(group, prefix)
                if v3 and comp == "B":
                    agg.update(aggregate_weighted(group, [1.0] * len(group), prefix))
                for k, v in agg.items():
                    if isinstance(v, (int, float)):
                        if not _KEEP_UNGATED.search(k):
                            keys[comp].add(k)
                        if v3 and comp == "B":
                            keys["B_all"].add(k)
    if v3:
        keys["B_all"] |= keys["B"] | {"openB_dose_sum_" + u for u in set(chemrep.UNIT_GROUPS.values())}
    return {c: tuple(sorted(v)) for c, v in keys.items()}


def jitter_rows(rows, cfg):
    """Augmented-copy jitter (label-preserving, domain common sense): temperatures + N(0, temp_sd) C, durations
    x 10**N(0, time_log_sd), additive doses x 10**N(0, dose_log_sd) (about +-10 %), and an acid strength strictly
    between 0 and 1 + N(0, acid_sd) (concentrated acids, strength ~0.9+, barely move).

    Deterministic per call (seeded by cfg["seed"] and the number of rows); has_* flags, unit strings and keys
    that are neither temperatures nor durations are unchanged; values keep their sign (no negative durations).
    """
    import numpy as np
    rng = np.random.default_rng([int(cfg.get("seed", 0)), len(rows)])
    sd_t, sd_l = float(cfg.get("temp_sd", 6.0)), float(cfg.get("time_log_sd", 0.08))
    sd_d, sd_a = float(cfg.get("dose_log_sd", 0.04)), float(cfg.get("acid_sd", 0.03))
    out = []
    for r in rows:
        x = dict(r)
        shift = rng.normal(0.0, sd_t)
        scale = 10.0 ** rng.normal(0.0, sd_l)
        dose_scale = 10.0 ** rng.normal(0.0, sd_d)
        for k, v in r.items():
            if k.startswith(("has_", "__")) or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            low = k.lower()
            if "temperature" in low and "unit" not in low:
                x[k] = float(v) + shift
            elif "duration" in low and "unit" not in low and v > 0:
                x[k] = float(v) * scale
            elif (low.endswith("_dose") or low.endswith("dose_volpct") or "_dose_sum_" in low) and v > 0:
                x[k] = float(v) * dose_scale
        if 0.0 < x.get("acid_post_present", 0.0) < 1.0:
            x["acid_post_present"] = min(1.0, max(0.0, x["acid_post_present"] + rng.normal(0.0, sd_a)))
        out.append(x)
    return out
