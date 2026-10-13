"""TRAIN-only scientific/word/context ML views; no IO, labels or physical law.

The inventory is candidate schema, never a globally fitted vocabulary. Historical
canonical values remain unchanged; unsupported whole units drop an extra family.
C2 alone accepts explicit paper context. Teacher4 is privileged NONNESTED input,
not an intrinsic physical state; caller supplies current columns[0,2,3,4].
"""
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import re
import unicodedata

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from . import features
from . import nonlinear_text_residual as core
VERSION, VIEWS = "fixed-scientific-word-context-views-v1", ("C0", "C1", "C2")
EXCLUDED = {"maximum_strain_on_pdms", "maximum_strain_on_prestrained_pdms", "tensile_strain"}
_VETO = re.compile(r"target|conductiv|resist|sigma|seebeck|power.?factor|carrier|mobility|"
    r"activation_energy|(?:^|_)t0(?:_|$)|vrh|^context_|^structure_|^cv_|"
    r"^(?:row_id|paper_id|source_id|sample|series|doi|review|evidence|association|result|sd)(?:_|$)", re.I)
_EXTRA = re.compile(r"^(?:has_|conflicting_)?extra_(.+?)(?:__(.*)|_unit)$")
_CAT = {
    "polymer_grade": ("ph1000", "ph500", "ai4083", "ai7580", "baytron_p", "sv3", "icp1020", "clevios_p"),
    "deposition": ("spin_coating", "blade_coating", "drop_casting", "casting", "screen_printing", "inkjet_printing", "spray_coating", "dip_coating", "wet_spinning", "electrospinning", "direct_ink_writing", "slot_die", "electrodeposition"),
    "substrate": ("glass", "silicon", "pet", "pen", "pdms", "pi", "ito", "fto", "metal", "paper", "textile", "free_standing"),
    "geometry": ("film", "fiber", "trace", "bulk", "gel", "porous", "vertical_junction"),
    "measurement": ("four_point", "two_point", "hall", "impedance", "dc", "in_plane", "out_of_plane"),
}
_DOSE = ("volpct", "wtpct", "molpct", "mmol_L", "mol_L", "mg_mL", "mL", "uL", "mg", "pct_unresolved", "amount_unresolved", "reported_ratio")
_DOSE_CAT = set(_DOSE) | {"unknown", "interval_or_bound", "ambiguous_multiple_values", "unparsed"}
_BASIS = {"unknown", "unresolved_denominator", "film_solids", "polymer_solids", "mixture", "stock_dispersion", "solution_molarity", "other_reported"}
_TIME = {"s", "sec", "seconds", "min", "minutes", "h", "hr", "hours", "d", "days"}
_TEMP = {"c", "°c", "degc", "celsius", "k", "kelvin"}
_CONC = {"m", "mm", "mol/l", "mmol/l", "mol l-1", "mmol l-1", "mg/ml", "g/l", "wt%", "wt.%", "vol%", "%"}
_UNKNOWN_UNITS = {"unknown", "unreported", "not reported", "unspecified"}


def _units(name):
    if "temperature" in name: return _TEMP
    if any(s in name for s in ("duration", "time")): return _TIME
    if "concentration" in name: return _CONC
    if name in {"ph", "solution_ph"}: return {"ph", "dimensionless", "1"}
    if name == "total_draw_ratio": return {"ratio", "dimensionless", "1"}
    if name == "spin_speed": return {"rpm", "r/min"}
    if name == "pump_speed": return {"ml/h", "ml/min", "ul/min", "µl/min", "μl/min"}
    if name == "laser_line_gap": return {"nm", "um", "µm", "μm", "mm"}
    if name == "relative_humidity": return {"%", "percent", "rh%"}
    if name == "measurement_frequency": return {"hz", "khz", "mhz", "ghz"}
    if name in {"applied_voltage", "ebl_acceleration_voltage"}: return {"v", "mv", "kv"}
    if name in {"organic_solvent_content", "final_ink_solids_wt_percent"}: return {"%", "wt%", "wt.%", "vol%", "vol.%"}
    if "speed" in name: return {"m/s", "mm/s", "cm/s", "mm/min", "m/min", "cm/min", "um/s", "µm/s", "μm/s"}
    return set()


def _unit(value):
    return unicodedata.normalize("NFKC", value).casefold().strip().replace("−", "-")


def configuration():
    return dict(version=VERSION, views=list(VIEWS), excluded_extra_families=sorted(EXCLUDED),
        extra_units={n:sorted(_units(n) | _UNKNOWN_UNITS) for n in sorted(features.RECIPE_EXTRA_NAMES-EXCLUDED)},
        categorical_combinations={k:list(v) for k,v in _CAT.items()}, teacher_columns=[0,2,3,4],
        canonical_scaling="none; TRAIN-observed scalar keys get explicit known flags",
        word_basis=core.configuration()["basis"], source_context="C2 only; TRAIN onehot; unseen zero",
        scope=__doc__.strip())


def _fixed():
    # Empty synthetic recipe + disabled descriptors has no registry/file access.
    empty = features.build_features({}, chemical_descriptors=False)
    names = set(empty) | set(features.NUMERIC_FIELDS) | {"measurement_temperature_K", "reported_ratio_basis"}
    names |= {f"{s}_duration_s" for s in ("dry", "anneal", "post_treatment")}
    names |= {"coating_temperature_C", "coating_speed_m_s", "post_temperature_C", "post_speed_m_s", "dmso_dose_volpct", "eg_dose_volpct"}
    for basis in ("mass", "molar", "basis_unreported"):
        names |= {f"{p}reported_pss_pedot_{basis}_ratio" for p in ("", "has_")}
    for chem in features.CHEMICAL_ALIASES:
        names.add(f"recipe_{chem}_dose_unit")
        for unit in _DOSE:
            names |= {f"{p}{stage}_{chem}_dose_{unit}" for p in ("", "has_") for stage in ("recipe", "post")}
        names |= {f"has_{stage}_{chem}_dose_{unit}" for stage in ("recipe", "post") for unit in _DOSE_CAT}
    for stage in ("recipe", "post"):
        names |= {f"{stage}_molecule_descriptor_count", f"{stage}_unmapped_chemical_count"}
        names |= {f"{stage}_molecule_{d}_{s}" for d in ("mol_wt", "mol_logp", "tpsa", "hbd", "hba") for s in ("mean", "min", "max")}
        names |= {f"has_{stage}_molecule_{d}_mean" for d in ("mol_wt", "mol_logp", "tpsa", "hbd", "hba")}
    return names


def make_schema(inventory):
    """Filter explicit inventory names/types, without rows, vocab fitting or IO."""
    if not isinstance(inventory, Mapping): raise ValueError("Inventory mapping required")
    num, cat = inventory.get("numeric_features"), inventory.get("categorical_features")
    if any(type(v) is not list or any(type(k) is not str or not k for k in v) or len(set(v))!=len(v) for v in (num,cat)) or set(num)&set(cat):
        raise ValueError("Unique disjoint numeric/categorical inventory names required")
    fixed, kept, excluded = _fixed(), {"numeric":[], "categorical":[]}, {}
    for kind, keys in (("numeric",num),("categorical",cat)):
        for key in sorted(keys):
            m = _EXTRA.fullmatch(key)
            if _VETO.search(key): reason = "forbidden_family"
            elif m:
                name, suffix = m.groups()
                if name not in features.RECIPE_EXTRA_NAMES or name in EXCLUDED: reason = "excluded_extra_family"
                elif kind != ("categorical" if key==f"extra_{name}_unit" else "numeric"): reason = "schema_type"
                elif suffix is not None and suffix not in {features._slug(u) for u in _units(name)|_UNKNOWN_UNITS}: reason = "unsupported_unit_slug"
                else: reason = None
            elif key not in fixed: reason = "outside_fixed_builder"
            elif kind != ("categorical" if key in _CAT or key in {"recipe_dose_basis", "reported_ratio_basis", "reported_ratio_direction"} or key.endswith("_dose_unit") else "numeric"): reason = "schema_type"
            else: reason = None
            if reason: excluded[key] = reason
            else: kept[kind].append(key)
    return dict(version=VERSION, numeric=kept["numeric"], categorical=kept["categorical"], excluded=excluded,
        inventory_counts=dict(numeric=len(num),categorical=len(cat),union=len(num)+len(cat)))


def _category(key,value):
    if type(value) is not str: raise ValueError("Canonical categorical string required")
    if key in _CAT:
        parts=value.split("+"); allowed=_CAT[key]
        ok=value in {"unknown","other_reported"} or parts==[x for x in allowed if x in parts]
    elif key=="recipe_dose_basis": ok=value in _BASIS
    elif key=="reported_ratio_basis": ok=value in {"mass","molar","basis_unreported"}
    elif key=="reported_ratio_direction": ok=value in {"unresolved","pss_to_pedot_explicit","pedot_to_pss_explicit","pedot_to_pss_field_order"}
    else: ok=value in _DOSE_CAT
    if not ok: raise ValueError("Outside closed canonical category")


def _project(rows,schema):
    nums, cats = set(schema["numeric"]), set(schema["categorical"])
    output=[]; counts=Counter()
    for row in rows:
        if not isinstance(row,Mapping): raise ValueError("Canonical feature mapping required")
        bad=set()
        families={m.group(1) for k in row if type(k) is str and (m:=_EXTRA.fullmatch(k)) and m.group(1) in features.RECIPE_EXTRA_NAMES-EXCLUDED}
        for name in families:
            key=f"extra_{name}_unit"
            if key not in row: bad.add(name); counts["missing_extra_unit_families"]+=1; continue
            unit=row[key]
            if type(unit) is not str: raise ValueError("Whole extra unit must be a string")
            unit=_unit(unit)
            if unit not in _units(name)|_UNKNOWN_UNITS: bad.add(name); counts["unsupported_extra_unit_families"]+=1; continue
            if any(m.group(2) is not None and m.group(2)!=features._slug(unit) for k in row if type(k) is str and (m:=_EXTRA.fullmatch(k)) and m.group(1)==name):
                bad.add(name); counts["mismatched_extra_unit_families"]+=1
        projected={}
        for key in row:
            if type(key) is not str: raise ValueError("String canonical keys required")
            if key not in nums|cats: counts["excluded_or_unknown_keys"]+=1; continue
            m=_EXTRA.fullmatch(key)
            if m and m.group(1) in bad: continue
            value=row[key]
            if key in nums:
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value):
                    raise ValueError("Finite nonboolean canonical scalar required")
                projected[key]=float(value)
            else:
                if m: value=_unit(value)  # already validated as a whole string above.
                else: _category(key,value)
                projected[key]=value
        output.append(projected)
    return output,counts


def _inputs(canonical,raw,teacher,sources,*,training=False):
    if any(isinstance(v,(Mapping,str,bytes)) for v in (canonical,raw,sources)): raise ValueError("Aligned row sequences required")
    canonical,raw,sources=list(canonical),list(raw),list(sources); n=len(canonical)
    a=np.asarray(teacher)
    if (len(raw)!=n or len(sources)!=n or (training and not n) or a.shape!=(n,4) or a.dtype.kind not in "iuf" or not np.isfinite(a).all()
            or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(teacher,dtype=object).flat)):
        raise ValueError("Aligned finite teacher4 and nonempty TRAIN required")
    if any(type(s) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}",s) is None for s in sources): raise ValueError("Explicit paper context strings required")
    return canonical,raw,a.astype(np.float64,copy=False),sources


def _matrices(basis,projected,raw,teacher,sources,counts,*,training=False):
    observed=set(basis["canonical_keys"]); nums=basis["numeric_keys"]; cats=set(basis["category_pairs"])
    rows=[]
    for row in projected:
        counts["query_only_keys"]+=len(set(row)-observed) if not training else 0
        counts["unseen_categories"]+=sum(isinstance(v,str) and (k,v) not in cats for k,v in row.items()) if not training else 0
        out={k:v for k,v in row.items() if k in observed}
        out.update({"known::"+k:float(k in row) for k in nums}); rows.append(out)
    vectorizer=basis["canonical_vectorizer"]
    canonical=vectorizer.fit_transform(rows) if training else (vectorizer.transform(rows) if rows else sparse.csr_matrix((0,len(vectorizer.feature_names_))))
    if training:
        text_basis,words,_=core.fit_basis(raw); basis["text_basis"]=text_basis
        source=basis["source_vectorizer"].fit_transform([{"paper_context":s} for s in sources])
    else:
        words,_=core.transform_basis(basis["text_basis"],raw)
        source=basis["source_vectorizer"].transform([{"paper_context":s} for s in sources]) if rows else sparse.csr_matrix((0,len(basis["source_vectorizer"].feature_names_)))
    counts["unseen_sources"]=sum(s not in basis["sources"] for s in sources)
    c0=sparse.hstack((canonical,sparse.csr_matrix(teacher)),format="csr")
    c1=sparse.hstack((c0,words),format="csr"); c2=sparse.hstack((c1,source),format="csr")
    designs=dict(zip(VIEWS,(c0,c1,c2)))
    if any(not np.isfinite(x.data).all() for x in designs.values()): raise ValueError("Finite sparse views required")
    audit=dict(n_rows=len(rows),counts=dict(counts),dimensions={k:x.shape[1] for k,x in designs.items()},
        canonical_numeric_keys=len(nums),canonical_encoded_width=canonical.shape[1],teacher_columns=4,
        word_numeric_width=words.shape[1],source_width=source.shape[1],
        candidate_schema_counts={k:len(basis["schema"][k]) for k in ("numeric","categorical")},
        schema_exclusion_counts=dict(Counter(basis["schema"]["excluded"].values())),
        observed_key_types={"numeric":len(nums),"categorical":len(observed)-len(nums)})
    return designs,audit


def fit_views(TRAINfeatures,TRAINraw,TRAINteacher4,TRAINsource,schema):
    """Return RAM-only basis, three CSR TRAIN designs and compact count audit."""
    if not isinstance(schema,Mapping) or schema.get("version")!=VERSION or set(schema)!={"version","numeric","categorical","excluded","inventory_counts"}:
        raise ValueError("Fixed inventory schema required")
    checked=make_schema(dict(numeric_features=schema["numeric"],categorical_features=schema["categorical"]))
    if checked["excluded"] or checked["numeric"]!=schema["numeric"] or checked["categorical"]!=schema["categorical"]:
        raise ValueError("Safe canonical schema required")
    features_,raw,teacher,sources=_inputs(TRAINfeatures,TRAINraw,TRAINteacher4,TRAINsource,training=True)
    projected,counts=_project(features_,schema); keys=sorted({k for row in projected for k in row})
    basis=dict(version=VERSION,schema=deepcopy(schema),canonical_keys=keys,numeric_keys=sorted(set(keys)&set(schema["numeric"])),
        category_pairs=sorted({(k,v) for row in projected for k,v in row.items() if isinstance(v,str)}),sources=sorted(set(sources)),
        canonical_vectorizer=DictVectorizer(sparse=True,sort=True),source_vectorizer=DictVectorizer(sparse=True,sort=True))
    designs,audit=_matrices(basis,projected,raw,teacher,sources,counts,training=True)
    return basis,designs,audit


def transform_views(basis,QUERYfeatures,QUERYraw,QUERYteacher4,QUERYsource):
    """QUERY-only frozen transform; no fit, vocabulary extension or label input."""
    if not isinstance(basis,Mapping) or basis.get("version")!=VERSION: raise ValueError("Fitted RAM basis required")
    features_,raw,teacher,sources=_inputs(QUERYfeatures,QUERYraw,QUERYteacher4,QUERYsource)
    projected,counts=_project(features_,basis["schema"])
    return _matrices(basis,projected,raw,teacher,sources,counts)
