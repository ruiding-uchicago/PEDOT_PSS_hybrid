"""Deterministic, information-preserving canonical wording for PEDOT:PSS experiment records.

    canonical_experiment(experiment) -> (experiment, notes)

The training corpus of the bundled model was rewritten with this function; predictions apply it unchanged.
Synonyms of one option map to one fixed wording; nothing the model's rule-based parsers or physics branches read is lost, because every candidate rewrite
is checked inside the function against the record's own unrewritten parse (so the guarantee also holds for unseen
inputs):

  * categorical fields (polymer_grade, deposition_method, substrate, specimen_geometry, measurement_method):
    candidates in order (1) the canonical wording of the categories the package parser finds (pedot_pss._categories),
    (2) an extended synonym group for frequent wordings the parser does not recognise (EXTENDED below),
    (3) the text with whitespace normalised; the first candidate that passes the check is used, otherwise the
    value is kept exactly as written;
  * additive_names, per name: (1) the CHEMICAL_NAMES wording if the name is a pure synonym of exactly one
    identified chemical (nothing left after removing that chemical's aliases, its abbreviation in parentheses,
    position/stereo locants and supplier words; names with a qualifier such as 'Zonyl FS-300', 'PEG-TmDD',
    'MWCNTs', 'Ag:PSS', 'DI water' are not pure), (2) a fixed spelling for a few qualified-name variant groups
    (QUALIFIED below), (3) the name with whitespace normalised;
  * all other fields (numbers, process_stages, extra_scalar_features, post_treatment, process_sequence, dose_basis,
    stock_solids_scope, amounts, ...) are never touched.

Check (signature = vendored feature_view(canonicalize(x)) + every branch admission with its coordinates: acid,
salt, PEO, SWCNT filler, pure-DAE, Polybot platform eligibility, P-layer chemistry/thermal eligibility). A rewrite
is accepted only if the signature equals the original one except for: identity features keyed by an additive name
(identity_recipe_*::<name>), the five string category labels, and listed category flags switching 0 -> 1.

Imports only the package's own modules (pedot_pss._categories and the vendored original parsers).
"""
from __future__ import annotations

import math
import re
import unicodedata

import numpy as np

from ._categories import CATEGORY_FIELDS, CHEMICAL_NAMES, parse, chemicals
from ._vendor import features as _features
from ._vendor.clean_inputs import canonicalize, feature_view, original_view
from ._vendor import original_special_physics as _osp
from ._vendor import original_filler_dae as _ofd
from ._vendor import original_physics_adapter as _opa
from ._vendor import original_polybot_clean as _opc
from ._vendor import acid_interface_closure as _acid

VERSION = "canonical-experiment-v1-1013"
CATEGORICAL = tuple(CATEGORY_FIELDS)
LABEL_KEYS = frozenset({"polymer_grade", "deposition", "substrate", "geometry", "measurement"})
FLAG_KEYS = frozenset(f"{prefix}_{category}" for prefix, table in CATEGORY_FIELDS.values() for category in table)
_MISSING = object()

# Extended synonym groups: (fullmatch pattern on the NFKC-lowercased, whitespace-collapsed text, fixed wording).
EXTENDED = {
    "polymer_grade": (
        (r"(?:heraeus\s+)?clevios(?:tm)?\s*(?:ph[\s\-]*)?1000", "PH1000"),
        (r"clevi(?:o|ou)s(?:tm)?\s*p(?:\s+aqueous\s+dispersion)?", "Clevios P"),
        (r"pt-?100(?:\s*\(nagase\s+chemtex(?:\s+corporation)?\))?(?:\s+aqueous\s+pedot\s*:\s*pss\s+dispersion)?",
         "PT-100 (Nagase ChemteX)"),
        (r"(?:agfa\s+)?orgacon\s+dry(?:\s*\(agfa\s+gevaert\s+n\.?v\.?\))?(?:,\s*lot\s+[a-z0-9 ]+)?", "Orgacon Dry (Agfa)"),
        (r"(?:dye|dry)\s+re-?dispersible\s+pedot\s*:\s*pss\s+pellets"
         r"|orgacon\s+water-soluble,\s*re-?dispersible\s+pedot\s*:\s*pss\s+pellets", "re-dispersible PEDOT:PSS pellets"),
        (r"(?:commercial\s+)?pedot\s*[:\-]\s*pss(?:\s+(?:aqueous\s+)?(?:dispersion|solution|suspension))?"
         r"|pedot\s*[:\-]\s*pss\s*\((?:unspecified|unknown|not\s+specified)(?:\s+grade)?\)", "PEDOT:PSS (unspecified grade)"),
        (r"(?:in[\s\-]*house[\s\-]*|self[\s\-]*|lab[\s\-]*)synthesi[sz]ed\s+pedot\s*[:\-]\s*pss", "in-house synthesized PEDOT:PSS"),
    ),
    "substrate": (
        (r"mylar(?:\s+(?:film|sheet|foil))?", "PET"),
        (r"(?:pre-cleaned\s+)?quartz(?:\s+(?:substrate|slide|plate))?", "quartz"),
        (r"(?:smooth,\s*flexible\s+)?polypropylene(?:\s*\(pp\))?(?:\s+(?:film|sheet|substrate))?"
         r"|pp(?:\s+(?:film|sheet|substrate))?", "polypropylene (PP)"),
        (r"(?:porous\s+)?pvdf(?:\s+durapore)?\s+membrane(?:[;,].*)?", "PVDF membrane"),
        (r"(?:polystyrene\s+)?petri\s+dish(?:;\s*free-?standing\s+film\s+after\s+drying)?", "petri dish"),
        (r"(?:none\s*[;:,]?\s*\(?\s*)?(?:self[\s\-]*(?:supporting|standing)|free[\s\-]*standing)"
         r"(?:\s+(?:film|fiber|fibre))?\)?(?:;\s*no\s+substrate\s+reported)?"
         r"|none;\s*film\s+peeled\s+from\s+beaker\s+bottom\s+and\s+self-supporting"
         r"|ptfe\s+beaker;\s*self-standing", "free-standing"),
    ),
    "deposition_method": (
        (r"(?:ultrasonic\s+)?aerosol\s+jet\s+printing(?:\s*\((?:ajp|aj®p)\))?", "aerosol jet printing"),
        (r"vacuum(?:-assisted)?\s+filtration(?:\s+in\s+(?:a\s+)?buchner\s+funnel)?", "vacuum filtration"),
        (r"drop[\s\-]*coating", "drop casting"),
        (r"lyophili[sz]ation(?:\s*\(freeze-drying\))?|freeze-drying", "freeze-drying"),
    ),
    "specimen_geometry": (),
    "measurement_method": (),
}
# Qualified additive names: fixed spelling per variant group (key = lowercase with spaces/hyphens/underscores removed).
QUALIFIED = {
    "zonylfs300": "Zonyl FS-300", "zonylfso100": "Zonyl FSO-100",
    "peg400": "PEG-400", "singlewalledcarbonnanotubes": "SWCNT", "swcnt": "SWCNT", "swcnts": "SWCNT",
    "multiwalledcarbonnanotubes(cnts)": "MWCNT", "multiwalledcarbonnanotubes": "MWCNT", "mwcnt": "MWCNT", "mwcnts": "MWCNT",
    "agnanowires(agnws)": "silver nanowires (AgNWs)", "silvernanowires(agnws)": "silver nanowires (AgNWs)",
}
# An alias match that is itself a qualified material form keeps the name qualified.
QUALIFIER_FORMS = (r"\b(?:mw|sw)cnts?\b", r"(?:multi|single)[\s\-]*walled", r"\brgo\b", r"graphene\s+oxide", r"\bgo\b",
                   r"nano(?:wire|particle|platelet|sheet|flake|rod)s?", r"\bnws?\b", r"\bpoly\s*\(", r":\s*pss\b")
SUPPLIER_WORDS = frozenset({"sigma", "aldrich", "merck", "alfa", "aesar", "tci", "acros", "fisher", "thermo", "fluka",
                            "honeywell", "macklin", "aladdin", "sinopharm", "vwr", "scientific", "chemicals", "co", "ltd",
                            "inc"})


def _ws(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _key(text):
    return _ws(text).lower().replace("−", "-")


def _squash(text):
    return re.sub(r"[\s\-_]+", "", _key(text))


def _same(a, b):
    if a is b:
        return True
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "fc" and b.dtype.kind in "fc":
            return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
        return bool(np.all(a == b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


# ----------------------------------------------------------------------------- signature and check
def signature(experiment):
    """(feature_view dict, admissions dict) of one experiment, as the model's parsers see it."""
    clean = canonicalize(experiment)
    fv = feature_view(clean)
    raw = original_view(clean)
    chain = canonicalize(raw)
    ai, si, pi = _osp.acid_inputs([chain]), _osp.salt_inputs([chain]), _osp.peo_inputs([chain])
    fi = _ofd.input_admission([raw])
    _, _, facts, _, chemistry, _ = _opa._project([raw])
    acid_row = ai["acid_rows"][0] or {}
    adm = {"acid": (bool(ai["protocol_admitted"][0]), acid_row.get(_acid.IDENTITY_KEY), acid_row.get(_acid.CONCENTRATION_KEY),
                    bool(ai["diagnostics"][0]["has_supported_typed_concentration"])),
           "salt": (bool(si["coordinates"]["active"][0]),) + tuple(float(si["coordinates"][k][0]) for k in
                    ("concentration_mol_L", "cation_softness", "contact_temperature_K", "contact_seconds")),
           "peo": (bool(pi["active"][0]), int(pi["family"][0]), float(pi["phi"][0]), float(pi["peo_mass"][0])),
           "filler": (bool(fi["admission_mask"][0]), float(fi["x"][0]), bool(fi["z"][0])),
           "dae": bool(fi["dae_mask"][0]),
           "platform": bool(_opc.eligible_scientific(fv)),
           "P_layer": (bool(chemistry[0]), bool(facts[0]["active"]))}
    return fv, adm


def violations(base, new):
    """Non-allowed differences between two signatures (empty list = information preserved)."""
    (fa, aa), (fb, ab) = base, new
    bad = [f"admission:{k}" for k in aa if not _same(aa[k], ab.get(k))]
    for k in set(fa) | set(fb):
        a, b = fa.get(k, _MISSING), fb.get(k, _MISSING)
        if a is not _MISSING and b is not _MISSING and _same(a, b):
            continue
        if k.startswith("identity_recipe_") and "::" in k:
            continue
        if k in LABEL_KEYS:
            continue
        if k in FLAG_KEYS and (a is _MISSING or a == 0.0) and b == 1.0:
            continue
        bad.append(k)
    return sorted(bad)


# ----------------------------------------------------------------------------- candidates
def category_candidates(field, value):
    out = []
    found = parse(field, value)
    if found:
        table = CATEGORY_FIELDS[field][1]
        out.append((" + ".join(table[c] for c in found), "parsed"))
    else:
        k = _key(value)
        for pattern, wording in EXTENDED.get(field, ()):
            if re.fullmatch(pattern, k):
                out.append((wording, "synonym"))
                break
    out.append((_ws(value), "whitespace"))
    seen, unique = set(), []
    for cand, kind in out:
        if cand not in seen:
            seen.add(cand); unique.append((cand, kind))
    return unique


def _is_abbreviation(token, context):
    """A parenthesised token whose letters occur in order in the surrounding name (e.g. 'gl' in 'glycerol')."""
    letters = [c for c in token if c.isalpha()]
    words = re.findall(r"[a-z]+", context)
    if not letters or not words or letters[0] not in {w[0] for w in words}:
        return False
    it = iter("".join(words))
    return all(c in it for c in letters)


def pure_synonym(name, key):
    """True if `name` names exactly the identified chemical `key` with nothing else (see module docstring)."""
    s = _features._text(name)
    if any(re.search(p, s) for p in QUALIFIER_FORMS):
        return False
    outside = re.sub(r"\([^)]*\)", " ", s)
    for pattern in _features.CHEMICAL_ALIASES[key]:
        s = re.sub(pattern, " ", s)
    s = s.replace(CHEMICAL_NAMES[key].lower(), " ")
    s = re.sub(r"\(\s*([a-z0-9+]{1,8})\s*\)", lambda m: " " if _is_abbreviation(m.group(1), outside) else m.group(0), s)
    s = re.sub(r"(?:^|(?<=[\s(]))[a-z0-9]{1,2}(?:,[a-z0-9]{1,2})*-(?=\s|\(|$)", " ", s)   # locants: 'n,n-', '3-', 'd-'
    tokens = [t for t in re.findall(r"[a-z0-9]+", s) if t not in SUPPLIER_WORDS]
    return not tokens


def additive_candidates(name):
    out = []
    keys = chemicals(name)
    if len(keys) == 1 and keys[0] in CHEMICAL_NAMES and pure_synonym(name, keys[0]):
        out.append((CHEMICAL_NAMES[keys[0]], "chemical"))
    q = QUALIFIED.get(_squash(name))
    if q:
        out.append((q, "qualified"))
    out.append((_ws(name), "whitespace"))
    seen, unique = set(), []
    for cand, kind in out:
        if cand not in seen:
            seen.add(cand); unique.append((cand, kind))
    return unique


# ----------------------------------------------------------------------------- the function
def canonical_experiment_detail(experiment):
    """(experiment, notes, events); events = [(field, old, new, kind, accepted, violations)]."""
    x0 = dict(experiment)
    base = signature(x0)
    cur, notes, events = dict(x0), [], []

    def attempt(field, old, cand, kind, trial):
        bad = violations(base, signature(trial))
        events.append((field, old, cand, kind, not bad, bad))
        if bad:
            notes.append(f"{field}: {old!r} -> {cand!r} refused ({kind}; would change {', '.join(bad[:3])})")
        else:
            notes.append(f"{field}: {old!r} -> {cand!r} ({kind})")
        return not bad

    for field in CATEGORICAL:
        value = cur.get(field)
        if not isinstance(value, str):
            continue
        for cand, kind in category_candidates(field, value):
            if cand == value:
                break
            trial = dict(cur); trial[field] = cand
            if attempt(field, value, cand, kind, trial):
                cur = trial
                break
    names = cur.get("additive_names")
    if isinstance(names, list):
        for idx in range(len(names)):
            name = cur["additive_names"][idx]
            if not isinstance(name, str):
                continue
            for cand, kind in additive_candidates(name):
                if cand == name:
                    break
                new_names = list(cur["additive_names"]); new_names[idx] = cand
                trial = dict(cur); trial["additive_names"] = new_names
                if attempt("additive_names", name, cand, kind, trial):
                    cur = trial
                    break
    final_bad = violations(base, signature(cur))
    if final_bad:                                   # cannot happen (each step is checked against the base)
        raise AssertionError(f"canonical rewrite lost information: {final_bad}")
    return cur, notes, events


def canonical_experiment(experiment):
    """Canonical wording of one experiment record; returns (experiment, notes)."""
    out, notes, _ = canonical_experiment_detail(experiment)
    return out, notes
