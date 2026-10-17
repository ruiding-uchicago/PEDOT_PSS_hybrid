"""Variant feature layer for the 'opened' PEDOT:PSS M/P hybrid (m-open, 1013 overnight study).

One transform, applied identically in training and prediction, to every place where chemistry/material
identity enters the ML parts of the hybrid:
  * the 5 base feature families returned by original_fallback._profiles (plain, identity_quantity, rich,
    state, enriched) -> wrapped `_profiles`;
  * the scientific feature view (clean_inputs.feature_view) used by the global kernel / late heads ->
    `transform_science` (the refit recomputes the scientific dicts with it) and a patched
    OriginalCleanSurrogate._project (platform admission computed from the UNtransformed view).

Components and modes (O = today, H = today + descriptors, D = descriptors only):
  A recipe additives (incl. fillers / nanomaterials), B post-treatment chemicals, C polymer grade, D substrate.
D removes the component's identity one-hots and alias-keyed features (see `removed`) and keeps:
  * GEM physics inputs (identity-based physics, left unchanged): dmso_dose_volpct, eg_dose_volpct (+has_),
    acid_post_present, polar_solvent_post_present, grade_ph1000/ph500/ai4083/baytron_p;
  * Polybot-platform keys (pb_*, cv_pb_*), the platform branch is identity-based by design;
  * generic counts / alignment / dose-basis keys and process-operation flags.
  `strict` D additionally hides the GEM keys from the feature dicts (proxy only: the GEM reads the same dict).
Physics branches (acid, salt, PEO, SWCNT filler, DAE, Polybot platform, P adapter) read raw rows and the
untransformed feature view; nothing here changes them.

Vendored from the m-open study: package-relative imports, no sys.path changes.
"""
from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path


from . import features as _F
from . import process_features as _PF
from . import reconstruction_features as _RF
from . import materials as _MAT

VERSION = "m-open-openrep-v1-1013"
COMPONENTS = ("A", "B", "C", "D")
ALIASES = tuple(_F.CHEMICAL_ALIASES)
PROC_CHEMS = tuple(sorted(set(_PF.CHEMICAL_PATTERNS) | {"acid"}, key=len, reverse=True))
RECON_POST = tuple(_RF._POST_CHEMICALS)
GEM_KEYS = {
    "A": ("dmso_dose_volpct", "has_dmso_dose_volpct", "eg_dose_volpct", "has_eg_dose_volpct"),
    "B": ("acid_post_present", "polar_solvent_post_present"),
    "C": ("grade_ph1000", "grade_ph500", "grade_ai4083", "grade_baytron_p"),
    "D": (),
}
_ALIAS_SORTED = tuple(sorted(ALIASES, key=len, reverse=True))
_A_EXACT = {"recipe_acid_present", "recipe_polar_solvent_present", "recipe_identified_chemical_count",
            "recipe_unknown_material_count", "recipe_unmapped_chemical_count", "recipe_molecule_descriptor_count"}
_B_EXACT = {"post_identified_chemical_count", "post_unmapped_chemical_count", "post_molecule_descriptor_count"}
_C_EXACT = {"polymer_grade", "grade_ai7580", "grade_icp1020", "grade_sv3", "recon_material_grade_label"}
_D_FLAGS = {"substrate"} | {"substrate_" + s for s in ("glass", "silicon", "pet", "pen", "pdms", "pi", "ito", "fto",
                                                       "metal", "paper", "textile", "free_standing")}
_PROC = re.compile(r"^(?:has_|ambiguous_|conflicting_)?proc_(pre_film|deposition|post|after_deposition|"
                   r"unspecified_stage|measurement)_(.+)_solution_concentration_.+$")
_EXTRA_B = re.compile(r"^(?:has_)?extra_(?:hz|tdae|tmu)_concentration_")


def _alias_after(rest):
    """Return the chemical alias that `rest` starts with (followed by '_'), else None."""
    for a in _ALIAS_SORTED:
        if rest.startswith(a + "_"):
            return a
    return None


@lru_cache(maxsize=200000)
def key_components(key):
    """Components whose D mode removes this feature key (frozenset; empty = never removed)."""
    k = key[4:] if key.startswith("has_") else key
    comps = set()
    if k in _A_EXACT or k.startswith(("recipe_molecule_", "identity_recipe_name::", "identity_recipe_dose::",
                                      "identity_recipe_dose_conflict::", "identity_recipe_has_dose::",
                                      "identity_recipe_family_")):
        comps.add("A")
    elif k.startswith("recipe_") and _alias_after(k[len("recipe_"):]):
        comps.add("A")
    if k in _B_EXACT or k.startswith(("post_molecule_", "recon_post_")) or key.startswith("conflicting_recon_post_"):
        comps.add("B")
    elif k.startswith("post_") and not k.startswith(("post_op_", "post_treatment", "post_temperature",
                                                       "post_speed")) and _alias_after(k[len("post_"):]):
        comps.add("B")
    m = _PROC.match(key)
    if m:
        stage = m.group(1)
        if stage in ("pre_film", "deposition"):
            comps.add("A")
        elif stage in ("post", "after_deposition"):
            comps.add("B")
        elif stage == "unspecified_stage":
            comps.update(("A", "B"))
    if _EXTRA_B.match(key):
        comps.add("B")
    if key in _C_EXACT or key.startswith("recon_vendor_"):
        comps.add("C")
    if key in _D_FLAGS:
        comps.add("D")
    return frozenset(comps)


def _gem_owner(key):
    for comp, keys in GEM_KEYS.items():
        if key in keys:
            return comp
    return None


_IDKEY = (
    (re.compile(r"^(?:has_)?recipe_(.+?)_(?:present|dose_.+)$"), "A", "alias"),
    (re.compile(r"^(?:has_)?post_(.+?)_(?:present|dose_.+)$"), "B", "alias"),
    (re.compile(r"^identity_recipe_name::(.+)$"), "A", "name"),
    (re.compile(r"^identity_recipe_dose::(.+?)::[^:]+::[^:]+$"), "A", "entity"),
    (re.compile(r"^identity_recipe_(?:dose_conflict|has_dose)::(.+)$"), "A", "entity"),
    (re.compile(r"^(?:has_|ambiguous_|conflicting_)?proc_(pre_film|deposition|post|after_deposition|unspecified_stage)_"
                r"(.+)_solution_concentration_"), None, "proc"),
    (re.compile(r"^(?:has_|conflicting_)?recon_post_(.+?)_(?:present|solution_.+)$"), "B", "recon"),
    (re.compile(r"^(?:has_)?extra_(hz|tdae|tmu)_concentration"), "B", "extra"),
)


@lru_cache(maxsize=200000)
def key_identity(key):
    """(kind, raw identity token) of a chemical-SPECIFIC key, or None (class-level/generic key)."""
    for pattern, comp, kind in _IDKEY:
        m = pattern.match(key)
        if not m:
            continue
        if kind == "alias":
            alias = _alias_after(m.group(1) + "_")
            return (kind, alias) if alias else None
        if kind == "proc":
            chem = m.group(2)
            return None if chem == "acid" else (kind, chem)
        return (kind, m.group(1))
    return None


class Spec:
    """modes: component -> 'O' | 'H' | 'D' | 'K<k>' (H with identity keys kept only for the top-k CID groups).

    strict: D also hides the GEM keys of D components (proxy only; the GEM reads the same dict).
    dropout: components whose identity keys are masked in an augmented training copy (identity dropout).
    topk: component -> list of CID groups kept in K modes (frozen in the spec; corpus input frequency).
    """

    def __init__(self, modes=None, strict=False, dropout=(), topk=None, brole=False, gate=None):
        modes = dict(modes or {})
        self.gate = dict(gate) if gate else None   # gating.py config (v0.2.1 audit fixes); None = v0.2.0 behaviour
        self.brole = bool(brole)          # B v2: rinse-clause chemicals kept apart from the active treatment
        self.modes = {c: modes.get(c, "O") for c in COMPONENTS}
        for c, m in self.modes.items():
            if m not in ("O", "H", "D") and not re.fullmatch(r"K\d+", m):
                raise ValueError(f"bad mode {c}={m}")
        self.strict = bool(strict)
        self.dropout = tuple(sorted(set(dropout)))
        self.topk = {c: list(v) for c, v in (topk or {}).items()}
        for c, m in self.modes.items():
            if m.startswith("K") and c not in self.topk:
                self.topk[c] = top_groups(c, int(m[1:]))
        self._topset = {c: set(v) for c, v in self.topk.items()}

    @property
    def tag(self):
        t = "".join(f"{c}{self.modes[c]}" for c in COMPONENTS)
        if self.dropout:
            t += "_drop" + "".join(self.dropout)
        if self.brole:
            t += "_brole"
        if self.gate:
            t += "_gate"
        return t + ("_strict" if self.strict else "")

    def is_identity(self):
        return all(m == "O" for m in self.modes.values())

    def describe(self):
        out = {"modes": dict(self.modes), "strict": self.strict, "dropout": list(self.dropout), "brole": self.brole,
               "topk": {c: v for c, v in self.topk.items() if self.modes.get(c, "O").startswith("K")},
               "version": VERSION}
        if self.gate:
            out["gate"] = dict(self.gate)
        return out

    def removed(self, key):
        comps = key_components(key)
        for c in comps:
            m = self.modes[c]
            if m == "D":
                return True
            if m.startswith("K"):
                ident = key_identity(key)
                if ident is not None and chem().identity_group(*ident) not in self._topset[c]:
                    return True
        if self.strict:
            owner = _gem_owner(key)
            return owner is not None and self.modes[owner] == "D"
        return False

    def masked(self, key):
        """Key hidden in the identity-dropout training copy (strict D for the dropout components)."""
        if not self.dropout:
            return False
        if key_components(key) & set(self.dropout):
            return True
        owner = _gem_owner(key)
        return owner is not None and owner in self.dropout


def top_groups(comp, k):
    """Top-k CID groups of component A or B (needs the training corpus; not available in the package)."""
    raise NotImplementedError("top-k identity modes need the training corpus; the package ships D modes")


_CHEM = None


def chem():
    """Lazy chemistry adapter (A/B descriptors); imported only when A or B is H/D."""
    global _CHEM
    if _CHEM is None:
        from . import chemrep
        _CHEM = chemrep.ChemRep()
    return _CHEM


def row_descriptors(canonical_row, spec):
    """Descriptor keys added for the H/D components of one canonical experiment row."""
    out = {}
    if spec.modes["C"] != "O":
        out.update(_MAT.grade_features(canonical_row.get("polymer_grade")))
    if spec.modes["D"] != "O":
        out.update(_MAT.substrate_features(canonical_row.get("substrate")))
    if spec.modes["A"] != "O":
        if spec.gate and spec.gate.get("v2"):
            from . import chemrep
            from . import gating
            out.update(chemrep.aggregate(gating.additive_items_v2(canonical_row, chem()), "openA_"))
        elif spec.gate and spec.gate.get("unify_units"):
            from . import chemrep
            from . import gating
            out.update(chemrep.aggregate(gating.unify_units(chem().additive_entities(canonical_row), canonical_row),
                                         "openA_"))
        else:
            out.update(chem().additive_features(canonical_row))
    if spec.modes["B"] != "O":
        out.update(chem().post_features(canonical_row, role_split=spec.brole))
    if spec.gate:
        from . import gating
        out.update(gating.gate_info(canonical_row, chem(), unify=bool(spec.gate.get("unify_units")),
                                    v2=bool(spec.gate.get("v2"))))
    return out


_KEYSPACE = None


def gate_keyspace():
    global _KEYSPACE
    if _KEYSPACE is None:
        from . import chemrep
        from . import gating
        _KEYSPACE = gating.keyspace(chem(), chemrep.SLOT_DESC, chemrep.CLASSES)
    return _KEYSPACE


def transform_dict(row, desc, spec):
    if spec.is_identity():
        return row
    out = {k: v for k, v in row.items() if not spec.removed(k)}
    out.update(desc)
    if spec.gate:
        from . import gating
        gating.apply(out, gate_keyspace(), spec.gate)
    return out


def mask_dict(row, spec):
    """Identity-dropout copy of an already transformed row (descriptors stay, identity keys go)."""
    return {k: v for k, v in row.items() if not spec.masked(k)}


def transform_profiles(canonical, profiles, spec):
    if spec.is_identity():
        return profiles
    descs = [row_descriptors(c, spec) for c in canonical]
    return {name: [transform_dict(r, d, spec) for r, d in zip(rows, descs)] for name, rows in profiles.items()}


def transform_science(canonical_rows, science_rows, spec):
    if spec.is_identity():
        return list(science_rows)
    return [transform_dict(s, row_descriptors(c, spec), spec) for c, s in zip(canonical_rows, science_rows)]


# ----------------------------------------------------------------------------- installation in the hybrid
_INSTALLED = {}


def install(spec):
    """Patch the hybrid (in this process) so every fit/predict sees the variant representation.

    * original_fallback._profiles -> transformed families (base learners, R07, GEM residual learners);
    * OriginalCleanSurrogate._project -> science transformed when computed internally; the platform
      admission `active` always from the untransformed feature view (physics routing unchanged).
    Idempotent; refuses a second, different spec.
    """
    if _INSTALLED:
        if _INSTALLED["spec"].describe() != spec.describe():
            raise RuntimeError("a different openrep spec is already installed")
        return status()
    import numpy as np
    from . import original_fallback as of
    from . import original_clean_surrogate as ocs
    from .clean_inputs import canonicalize, original_view, feature_view
    from .original_polybot_clean import eligible_scientific
    original_profiles = of._profiles

    def _profiles_open(rawrows):
        canonical, profiles = original_profiles(rawrows)
        return canonical, transform_profiles(canonical, profiles, spec)

    def _project_open(X, scientific=None):
        clean = [canonicalize(row) for row in X]
        raw = [original_view(row) for row in clean]
        untransformed = [feature_view(row) for row in clean]
        if scientific is None:
            science = transform_science(clean, untransformed, spec)
        else:
            science = list(scientific)
        if len(science) != len(raw):
            raise ValueError("Scientific rows must correspond exactly to experiments")
        active = np.asarray([not eligible_scientific(row) for row in untransformed], dtype=bool)
        return clean, raw, science, active

    _profiles_open.openrep_spec = spec.describe()
    of._profiles = _profiles_open
    ocs.OriginalCleanSurrogate._project = staticmethod(_project_open)
    _INSTALLED.update(spec=spec, original_profiles=original_profiles)
    return status()


def status():
    if not _INSTALLED:
        return {"openrep_installed": False}
    return {"openrep_installed": True, "openrep_version": VERSION, **_INSTALLED["spec"].describe()}


def spec_from_payload(payload):
    return Spec(payload.get("modes"), payload.get("strict", False), payload.get("dropout", ()), payload.get("topk"),
                payload.get("brole", False), payload.get("gate"))


def spec_from_env(default=None):
    """OPENREP_SPEC='{"modes": {...}, ...}' or a path to such a JSON file."""
    raw = os.environ.get("OPENREP_SPEC")
    if not raw:
        return default
    payload = json.loads(Path(raw).read_text()) if not raw.lstrip().startswith("{") else json.loads(raw)
    return spec_from_payload(payload)
