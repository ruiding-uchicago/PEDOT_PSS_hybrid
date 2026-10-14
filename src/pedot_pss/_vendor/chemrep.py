"""Identity-free chemistry representation of components A (recipe additives) and B (post-treatment chemicals).

Input: the bundled chemistry table (chem_table.json, resolved through PubChem): for each corpus additive
name and each post-treatment chemical key a resolved structure (SMILES, CID synonym group), 35 RDKit
descriptors (14 molecular + 21 SMARTS functional-group counts), class block (polymer / nanomaterial / vague /
mixture). Post-treatment chemicals are detected in features._text(post_treatment) with the table's regex.

Per row and component the representation is identity-free and dose-aware:
  * counts: present entities, with structure, polymers, nanomaterials, unresolved;
  * dose: each entity's parsed dose (features._parse_dose for additives; clause-local concentration
    association for post-treatment text) -> unit-normalised saturation s = d/(d + h_unit)
    (h = 5 vol%/wt%/mol%, 0.1 mol/L, 10 mg/mL; unknown dose -> s = 0.5), per-unit totals;
  * structure: s-weighted means of the 35 descriptors, min/max of MW/logP/TPSA, and two slots (the two
    entities with the largest s: 18 key descriptors + dose value/unit/s each), so per-molecule dose survives;
  * generic classes derived from STRUCTURE (RDKit SMARTS rules below, any molecule): acid, polar solvent,
    organic-cation ionic liquid / organic salt, metal salt, surfactant, base; plus polymer / nanomaterial
    (carbon, metal, 1D, 2D) from the class block; each as presence flag and s-weighted dose sum.
CID synonym groups are merged (one entity per group_key per row).

Vendored from the m-open study: package-relative imports; listed names use precomputed class flags.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from . import openrep as R
from . import features as _F
from . import process_features as _PF

TABLE = Path(__file__).with_name("chem_table.json")      # frozen m-chem v1.3, reduced (export_open_chem.py)
LOG_DESC = {"mol_wt", "heavy_atom_count", "mol_mr"}
CORE = ("mol_wt", "mol_logp", "tpsa", "hbd", "hba", "num_rotatable_bonds", "fraction_csp3", "num_aromatic_rings",
        "heavy_atom_count", "mol_mr", "formal_charge", "fragment_count", "positive_charge_count",
        "negative_charge_count")
SLOT_DESC = ("mol_wt", "mol_logp", "tpsa", "hbd", "hba", "num_rotatable_bonds", "fraction_csp3",
             "num_aromatic_rings", "heavy_atom_count", "positive_charge_count", "negative_charge_count",
             "fg_sulfonic_acid_or_sulfonate", "fg_sulfate_group", "fg_sulfoxide", "fg_hydroxyl",
             "fg_carboxylic_acid_or_carboxylate", "fg_ether", "fg_amide")
CLASSES = ("acid", "polar_solvent", "ionic_liquid", "metal_salt", "surfactant", "base", "polymer",
           "nano_carbon", "nano_metal", "nano_1d", "nano_2d", "unresolved")
HALF = {"volpct": 5.0, "wtpct": 5.0, "molpct": 5.0, "pct_unresolved": 5.0, "pct_basis_unresolved": 5.0,
        "mol_L": 0.1, "normality_eq_L": 0.1, "mg_mL": 10.0, "reported_ratio": 0.05}
UNIT_GROUPS = {"volpct": "volpct", "wtpct": "wtpct", "mol_L": "molL", "mmol_L": "molL", "normality_eq_L": "molL",
               "mg_mL": "mgmL", "pct_unresolved": "pct", "pct_basis_unresolved": "pct", "molpct": "pct"}
_EMPTY = {"", "none", "no additive", "pristine", "untreated"}
METALS = {"Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Sb",
          "Te", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ag", "Au", "Pt", "Pd", "Zr", "Mo", "W", "Cd",
          "Hg", "Ce", "La", "Y", "Sc", "Nb", "Ta", "Ru", "Rh", "Ir", "Os"}
_EXTRA_POST = {"hz_concentration": "hydrazine", "tdae_concentration": "tdae", "tmu_concentration": "tmu"}


# ----------------------------------------------------------------------------- structure-derived classes
@lru_cache(maxsize=4096)
def structure_classes(smiles):
    """Generic class flags computed from a SMILES string (works for any new molecule)."""
    out = {c: 0.0 for c in ("acid", "polar_solvent", "ionic_liquid", "metal_salt", "surfactant", "base")}
    if not smiles:
        return out
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors
    mol = Chem.MolFromSmiles(smiles.replace("*", "[H]"), sanitize=True)
    if mol is None:
        return out
    atoms = list(mol.GetAtoms())
    symbols = {a.GetSymbol() for a in atoms}
    has_metal = bool(symbols & METALS)
    charged = any(a.GetFormalCharge() != 0 for a in atoms)
    frags = Chem.GetMolFrags(mol, asMols=True)
    P = lambda s: Chem.MolFromSmarts(s)
    # acid: O-H on an oxo-acid centre (C(=O), S, P, N+, Cl, B, Se) or a hydrogen halide molecule
    acid = mol.HasSubstructMatch(P("[CX3](=O)[OX2H1]")) or mol.HasSubstructMatch(P("[S,P,Se,Cl,B,$([N+])][OX2H1]"))
    for f in frags:
        if f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() in ("F", "Cl", "Br", "I") \
                and f.GetAtomWithIdx(0).GetFormalCharge() == 0:
            acid = True
    out["acid"] = float(acid)
    out["metal_salt"] = float(has_metal)
    # organic-cation salt / ionic liquid: a positively charged N/P/S fragment with >= 3 carbons, no metal
    if not has_metal:
        for f in frags:
            pos = any(a.GetFormalCharge() > 0 and a.GetSymbol() in ("N", "P", "S") for a in f.GetAtoms())
            neg_inner = any(a.GetFormalCharge() < 0 for a in f.GetAtoms())
            ncarb = sum(1 for a in f.GetAtoms() if a.GetSymbol() == "C")
            if pos and ncarb >= 3 and (len(frags) > 1 or not neg_inner):
                out["ionic_liquid"] = 1.0
    # polar solvent: one neutral fragment, no metal, MW <= 200, has N/O/S, Crippen logP <= 1.0, not an acid
    if len(frags) == 1 and not charged and not has_metal and not acid and "*" not in smiles:
        mw = Descriptors.MolWt(mol)
        if mw <= 200 and symbols & {"N", "O", "S"} and Crippen.MolLogP(mol) <= 1.0:
            out["polar_solvent"] = 1.0
    # surfactant: hydrophobic tail (>= 8 aliphatic C in a row, perfluoro C4, or tert-octylphenyl)
    # + hydrophilic head (charge, sulfonate/sulfate/carboxyl/phosphate, or >= 2 ether/hydroxyl O)
    tail = (mol.HasSubstructMatch(P("[CX4,CX3;!R][CX4,CX3;!R][CX4,CX3;!R][CX4,CX3;!R][CX4,CX3;!R][CX4,CX3;!R]"
                                    "[CX4,CX3;!R][CX4,CX3;!R]"))
            or mol.HasSubstructMatch(P("[CX4](F)(F)[CX4](F)(F)[CX4](F)(F)[CX4](F)"))
            or mol.HasSubstructMatch(P("CC(C)(C)CC(C)(C)c1ccccc1")))
    if tail:
        n_o = len(mol.GetSubstructMatches(P("[OX2;$([OX2]([#6])[#6]),$([OX2H1][#6])]")))
        head = (charged or n_o >= 2 or mol.HasSubstructMatch(P("[SX4](=O)(=O)[O,N]"))
                or mol.HasSubstructMatch(P("[CX3](=O)[OX1,OX2H1]")) or mol.HasSubstructMatch(P("P(=O)[O]")))
        out["surfactant"] = float(bool(head))
    # base: hydroxide anion, ammonia/aliphatic amine, amidine (not amide/sulfonamide/aromatic amine)
    base = mol.HasSubstructMatch(P("[OX1H1-]")) or smiles.strip() == "N"
    if not acid and not base:
        base = (mol.HasSubstructMatch(P("[NX3;!$(N[#6]=[O,S,N]);!$(N[S,P]=O);!$(Na);!$([N+]);!$(N[N+])]"))
                or mol.HasSubstructMatch(P("[NX2;!R0,R]=[CX3][NX3]")))
        base = base and not charged
    out["base"] = float(bool(base))
    return out


# ----------------------------------------------------------------------------- table adapter
class ChemRep:
    def __init__(self, path=TABLE):
        self.path = Path(path)
        self.table = json.loads(self.path.read_text())
        self.names = self.table["names"]
        self.post = self.table["post_treatment_chemicals"]
        self._post_patterns = []
        self._post_negation = {}
        for key, rec in self.post.items():
            pat = rec.get("pattern")
            if pat:
                try:
                    self._post_patterns.append((key, re.compile(pat)))
                except re.error:
                    pass
            neg = rec.get("negation_pattern")
            if neg:
                try:
                    self._post_negation[key] = re.compile(neg)
                except re.error:
                    pass
        self._entity = {}
        self._build_identity_maps()

    # -- identity token (from a chemical-specific legacy feature key) -> CID group
    _PROC_TO_POST = {"mai": "methylammonium_iodide", "mabr": "methylammonium_bromide", "nai": "sodium_iodide",
                     "gbl": "gamma_butyrolactone", "emitfsi": "emim_tfsi", "camphorsulfonic_acid_label":
                     "camphorsulfonic_acid", "litfsi": "lithium_tfsi", "hz": "hydrazine"}

    def _build_identity_maps(self):
        from .identity_features import normalize_identity_name, _DOSE_ALIASES
        from collections import Counter
        self._name_group = {}
        for name, rec in self.names.items():
            norm = normalize_identity_name(name)
            if norm is not None and norm not in self._name_group:
                self._name_group[norm] = self.entity(rec, name)["group"]
        self._entity_group = {}
        for ent, aliases in _DOSE_ALIASES.items():
            for a in aliases:
                if a in self._name_group:
                    self._entity_group[ent] = self._name_group[a]
                    break
        # alias -> group: post table entry, else the most frequent corpus name the alias regex matches
        self._alias_group = {}
        cnt = Counter()
        for name, rec in self.names.items():
            chems = _F._chemicals(name)
            if len(chems) == 1:
                cnt[(chems[0], self.entity(rec, name)["group"])] += int(rec.get("mentions") or 1)
        for alias in _F.CHEMICAL_ALIASES:
            if alias in self.post:
                self._alias_group[alias] = self.entity(self.post[alias], alias)["group"]
            else:
                best = sorted(((c, g) for (a, g), c in cnt.items() if a == alias), reverse=True)
                self._alias_group[alias] = best[0][1] if best else "alias:" + alias

    def identity_group(self, kind, token):
        if kind == "alias":
            return self._alias_group.get(token, "alias:" + token)
        if kind == "name":
            return self._name_group.get(token, "name:" + token)
        if kind == "entity":
            return self._entity_group.get(token) or self._name_group.get(token, "name:" + token)
        if kind in ("proc", "recon", "extra"):
            key = self._PROC_TO_POST.get(token, token)
            if key in self.post:
                return self.entity(self.post[key], key)["group"]
            if key in self._alias_group:
                return self._alias_group[key]
            return "post:" + key
        return kind + ":" + token

    # -- one resolved entity -> (group, descriptor dict, class dict)
    def entity(self, rec, fallback_key):
        cache_key = id(rec) if rec is not None else ("none", fallback_key)
        if cache_key in self._entity:
            return self._entity[cache_key]
        if rec is None:
            ent = {"group": "unres:" + fallback_key, "desc": None, "polydesc": None,
                   "cls": {**{c: 0.0 for c in CLASSES}, "unresolved": 1.0}, "struct": False}
            self._entity[cache_key] = ent
            return ent
        rd = rec.get("rdkit")
        smiles = rec.get("smiles_descriptor_form") or rec.get("smiles")
        klass = rec.get("class") or ""
        cls = {c: 0.0 for c in CLASSES}
        cls.update(rec["cls_rdkit"] if "cls_rdkit" in rec else
                   (structure_classes(smiles) if smiles else {}))   # listed names: precomputed
        if klass == "polymer":
            cls["polymer"] = 1.0
            cls["polar_solvent"] = 0.0
        nano = rec.get("nanomaterial") or {}
        if klass == "nanomaterial":
            mat = str(nano.get("nominal_material", "")).lower()
            cls["nano_carbon"] = float(bool(nano.get("carbon_based")))
            cls["nano_metal"] = float(any(m in mat for m in ("silver", "gold", "copper", "nickel", "platinum",
                                                            "metal", "mxene", "ti3c2")))
            dim = str(nano.get("dimensionality", ""))
            cls["nano_1d"] = float(dim.startswith("1"))
            cls["nano_2d"] = float(dim.startswith("2"))
        if klass == "vague":
            text = str(rec.get("certain_class", "")).lower()
            cls["surfactant"] = float("surfactant" in text)
            cls["polar_solvent"] = float("solvent" in text)
            cls["unresolved"] = 1.0
        if klass == "mixture_or_commercial_product" and not rd:
            text = (str(rec.get("product_type", "")) + " " + str(rec.get("canonical_name", ""))).lower()
            cls["surfactant"] = float("surfactant" in text or "wetting" in text)
            cls["unresolved"] = 1.0
        def conv(rdkit):
            out = {}
            for k, v in (rdkit or {}).items():
                if v is None:
                    continue
                v = float(v)
                if k in LOG_DESC:
                    v = math.log10(max(v, 1.0))
                out[k] = v
            return out
        desc, polydesc = None, None
        poly = rec.get("polymer") or {}
        if klass == "polymer":
            olig = (poly.get("oligomer") or {}).get("rdkit")
            if olig:                                   # explicit chain for short PEG/PEGDA: a real molecule
                desc = conv(olig)
            elif rd:                                   # repeat unit only: kept apart from small molecules
                polydesc = conv(rd)
            mn = poly.get("nominal_Mn")
            if isinstance(mn, (int, float)) and mn > 0:
                polydesc = dict(polydesc or {}, log10_mn=math.log10(mn))
        elif rd:
            desc = conv(rd)
        ent = {"group": rec.get("group_key") or ("name:" + fallback_key), "desc": desc, "polydesc": polydesc,
               "cls": cls, "struct": desc is not None}
        self._entity[cache_key] = ent
        return ent

    # -- additives (A)
    def additive_entities(self, row):
        names = row.get("additive_names") or []
        amounts = row.get("additive_amounts_raw") or []
        if not isinstance(names, list):
            names = [names]
        if not isinstance(amounts, list):
            amounts = [amounts]
        aligned = len(names) == len(amounts)
        basis = row.get("dose_basis")
        merged = {}
        for i, name in enumerate(names):
            if not isinstance(name, str) or _F._text(name) in _EMPTY:
                continue
            dose, unit = _F._parse_dose(amounts[i], basis) if aligned else (None, "unknown")
            if dose is not None and dose == 0:
                continue
            ent = self.entity(self.names.get(name), name)
            g = ent["group"]
            if g in merged:                      # CID synonym group: one entity, keep the larger dose
                if dose is not None and (merged[g]["dose"] is None or dose > merged[g]["dose"]):
                    merged[g].update(dose=dose, unit=unit)
                continue
            merged[g] = {"ent": ent, "dose": dose, "unit": unit}
        return list(merged.values())

    def additive_identities(self, row):
        return [e["ent"]["group"] for e in self.additive_entities(row)]

    # -- post-treatment chemicals (B)
    def post_entities(self, row):
        text = _F._text(row.get("post_treatment"))
        if re.fullmatch(r"(?:none|no(?:ne)?(?: post[- ]treatment)?|untreated|not applicable|n/a)", text):
            text = ""
        found = {}
        if text:
            hits = [(key, m.start(), m.end()) for key, pat in self._post_patterns for m in pat.finditer(text)
                    if not (key in self._post_negation and self._post_negation[key].search(text))]
            # v2: concentrations are read from the case-preserved text (molarity 'M' vs 'mM' is case-sensitive);
            # _F._text only lower-cases, so offsets agree whenever the lengths agree.
            cased = unicodedata.normalize("NFKC", str(row.get("post_treatment"))).strip().replace("\u2212", "-")
            conc = self._concentrations(cased if len(cased) == len(text) else text, hits)
            # clause role: a mention inside a rinse/wash/cleaning clause is a rinse, not the active treatment.
            # v2: ', then ...', '... then ...', 'followed by', and a comma/'and' right before a rinse verb also
            # start a new clause ('soaked in H2SO4, then washed with water': H2SO4 active, water rinse).
            cuts = [m.end() for m in re.finditer(r";|→|->", text)]
            cuts += [m.start() for m in re.finditer(
                r",\s*(?:and\s+)?(?:then|subsequently|followed\s+by|after\s+which)\b|\s(?:and\s+)?then\s|"
                r"\bfollowed\s+by\b|,\s*(?=(?:and\s+)?(?:rins|wash|clean))|\band\s+(?=(?:rins|wash|clean))", text)]
            bounds = sorted({0, *cuts, len(text) + 1})
            clauses = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
            for key, start, _ in hits:
                ent = self.entity(self.post.get(key), key)
                g = ent["group"]
                dose, unit = conc.get(key, (None, "unknown"))
                lo, hi = next(((a, b) for a, b in clauses if a <= start < b), (0, len(text)))
                rinse = bool(re.search(r"rins|wash|clean|remov", text[lo:hi]))
                if g in found:
                    if dose is not None and found[g]["dose"] is None:
                        found[g].update(dose=dose, unit=unit)
                    found[g]["rinse"] = found[g]["rinse"] and rinse
                    continue
                found[g] = {"ent": ent, "dose": dose, "unit": unit, "rinse": rinse}
        for item in row.get("extra_scalar_features") or []:
            key = _EXTRA_POST.get(str(item.get("name")))
            if not key or item.get("value") is None:
                continue
            ent = self.entity(self.post.get(key), key)
            g = ent["group"]
            unit = "volpct" if str(item.get("unit", "")).replace(" ", "").lower() in ("vol%", "%") else "pct_unresolved"
            if g in found:
                if found[g]["dose"] is None:
                    found[g].update(dose=float(item["value"]), unit=unit)
                found[g]["rinse"] = False
            else:
                found[g] = {"ent": ent, "dose": float(item["value"]), "unit": unit, "rinse": False}
        # explicit zero concentration = absent
        return [v for v in found.values() if not (v["dose"] is not None and v["dose"] == 0)]

    def post_identities(self, row):
        return [e["ent"]["group"] for e in self.post_entities(row)]

    @staticmethod
    def _concentrations(text, hits):
        """Clause-local association of explicit concentrations with chemical mentions (process_features rule)."""
        out = {}
        connectors = {"of", "in", "at", "aqueous", "solution", "bath", "concentration", "about", "approximately",
                      "by", "volume", "weight", "mass"}
        for m in _PF.CONCENTRATION.finditer(text):
            if not _PF._valid_scalar(text, m):
                continue
            choices = []
            for key, s, e in hits:
                between = text[m.end():s] if s >= m.end() else text[e:m.start()]
                distance = max(s - m.end(), m.start() - e, 0)
                words = set(re.findall(r"[a-z]{2,}", between.lower()))
                if distance <= 28 and words <= connectors and not re.search(r"[,;→]|\b(?:then|and)\b|->", between, re.I):
                    choices.append((distance, key))
            choices.sort()
            if not choices or (len(choices) > 1 and choices[0][0] == choices[1][0] and choices[0][1] != choices[1][1]):
                continue
            unit, factor = _PF._unit(m.group("unit"))
            value = float(m.group("value")) * factor
            key = choices[0][1]
            if key not in out and math.isfinite(value):
                out[key] = (value, unit)
        return out

    # -- features
    def additive_features(self, row):
        return aggregate(self.additive_entities(row), "openA_")

    def post_features(self, row, role_split=False):
        items = self.post_entities(row)
        if not role_split:
            return aggregate(items, "openB_")
        active = [e for e in items if not e.get("rinse")]
        rinse = [e for e in items if e.get("rinse")]
        out = aggregate(active, "openB_")
        out["openB_rinse_n"] = float(len(rinse))
        for c in ("polar_solvent", "acid", "base", "metal_salt"):
            if any(e["ent"]["cls"].get(c) for e in rinse):
                out["openB_rinse_cls_" + c] = 1.0
        vals = [e["ent"]["desc"] for e in rinse if e["ent"]["desc"]]
        for k in ("mol_wt", "mol_logp", "tpsa", "hbd"):
            v = [d[k] for d in vals if k in d]
            if v:
                out["openB_rinse_mean_" + k] = float(sum(v) / len(v))
        return out


def saturation(dose, unit):
    if dose is None or not math.isfinite(dose) or dose < 0:
        return None
    if unit == "mmol_L":
        dose, unit = dose / 1000.0, "mol_L"
    h = HALF.get(unit)
    if h is None:
        return None
    return dose / (h + dose)


def aggregate(items, p):
    out = {p + "n": float(len(items))}
    if not items:
        return out
    out[p + "n_struct"] = float(sum(e["ent"]["struct"] for e in items))
    sats = []
    unit_sum = {}
    for e in items:
        s = saturation(e["dose"], e["unit"])
        e["s"] = s
        sats.append(0.5 if s is None else s)
        if e["dose"] is not None and e["unit"] in UNIT_GROUPS:
            u = UNIT_GROUPS[e["unit"]]
            d = e["dose"] / 1000.0 if e["unit"] == "mmol_L" else e["dose"]
            unit_sum[u] = unit_sum.get(u, 0.0) + d
    out[p + "n_dose_known"] = float(sum(e["s"] is not None for e in items))
    out[p + "sat_sum"] = float(sum(e["s"] for e in items if e["s"] is not None))
    for u, v in unit_sum.items():
        out[p + "dose_sum_" + u] = v
    for c in CLASSES:
        flags = [e["ent"]["cls"].get(c, 0.0) for e in items]
        if any(flags):
            out[p + "cls_" + c] = 1.0
            out[p + "clsdose_" + c] = float(sum(f * w for f, w in zip(flags, sats)))
    struct = [(e, w) for e, w in zip(items, sats) if e["ent"]["desc"]]
    if struct:
        keys = sorted({k for e, _ in struct for k in e["ent"]["desc"]})
        tw = sum(w for _, w in struct) or 1.0
        for k in keys:
            vals = [(e["ent"]["desc"][k], w) for e, w in struct if k in e["ent"]["desc"]]
            if vals:
                out[p + "wm_" + k] = float(sum(v * w for v, w in vals) / max(sum(w for _, w in vals), 1e-9))
        for k in ("mol_wt", "mol_logp", "tpsa"):
            vals = [e["ent"]["desc"][k] for e, _ in struct if k in e["ent"]["desc"]]
            if vals:
                out[p + "max_" + k] = float(max(vals))
                out[p + "min_" + k] = float(min(vals))
    polys = [(e, w) for e, w in zip(items, sats) if e["ent"].get("polydesc")]
    if polys:
        for k in ("mol_logp", "tpsa", "hbd", "hba", "fg_ether", "fg_hydroxyl", "fg_sulfonic_acid_or_sulfonate",
                  "fg_amide", "fg_amine", "fg_ester", "positive_charge_count", "negative_charge_count", "log10_mn"):
            vals = [(e["ent"]["polydesc"][k], w) for e, w in polys if k in e["ent"]["polydesc"]]
            if vals:
                out[p + "poly_wm_" + k] = float(sum(v * w for v, w in vals) / max(sum(w for _, w in vals), 1e-9))
    order = sorted(range(len(items)), key=lambda i: (-sats[i], items[i]["ent"]["group"]))
    for slot, i in enumerate(order[:2], start=1):
        e = items[i]
        q = f"{p}s{slot}_"
        out[q + "sat"] = float(sats[i])
        if e["dose"] is not None:
            out[q + "dose"] = float(e["dose"])
        out[q + "unit"] = str(e["unit"])
        d = e["ent"]["desc"]
        if d:
            for k in SLOT_DESC:
                if k in d:
                    out[q + k] = d[k]
        pd = e["ent"].get("polydesc")
        if pd:
            for k in ("mol_logp", "tpsa", "fg_ether", "fg_hydroxyl", "fg_sulfonic_acid_or_sulfonate", "log10_mn"):
                if k in pd:
                    out[q + "poly_" + k] = pd[k]
        for c in ("acid", "polar_solvent", "ionic_liquid", "metal_salt", "surfactant", "polymer", "nano_carbon",
                  "nano_metal"):
            if e["ent"]["cls"].get(c):
                out[q + "cls_" + c] = 1.0
    return out
