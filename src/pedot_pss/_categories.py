"""Fixed-option (one-hot) wording for the categorical fields.

The model parses these text fields into fixed categories with its own rules. ``strict`` mode runs that
same parser, then replaces the text by one canonical wording per category, so synonyms and rephrasings
of the same option give identical predictions. Text that maps to no category is rejected with the list
of allowed options.
"""
from __future__ import annotations

from functools import lru_cache
import re

from ._vendor import features as _features
from ._vendor.clean_inputs import canonicalize, feature_view

# field -> (feature prefix, {category: canonical wording})
CATEGORY_FIELDS = {
    "polymer_grade": ("grade", {
        "ph1000": "PH1000", "ph500": "PH500", "ai4083": "AI4083", "ai7580": "AI7580",
        "baytron_p": "Baytron P", "icp1020": "ICP1020", "sv3": "SV3", "clevios_p": "Clevios P"}),
    "deposition_method": ("deposition", {
        "spin_coating": "spin coating", "blade_coating": "blade coating", "drop_casting": "drop casting",
        "casting": "casting", "screen_printing": "screen printing", "inkjet_printing": "inkjet printing",
        "spray_coating": "spray coating", "dip_coating": "dip coating", "wet_spinning": "wet spinning",
        "electrospinning": "electrospinning", "direct_ink_writing": "direct ink writing",
        "slot_die": "slot-die coating", "electrodeposition": "electrodeposition"}),
    "substrate": ("substrate", {
        "glass": "glass", "silicon": "silicon", "pet": "PET", "pen": "PEN", "pdms": "PDMS", "pi": "polyimide",
        "ito": "ITO", "fto": "FTO", "metal": "metal (gold)", "paper": "paper", "textile": "textile",
        "free_standing": "free-standing"}),
    "specimen_geometry": ("geometry", {
        "film": "film", "fiber": "fiber", "trace": "printed trace", "bulk": "bulk", "gel": "gel",
        "porous": "porous", "vertical_junction": "vertical junction"}),
    "measurement_method": ("measurement", {
        "four_point": "four-point probe", "two_point": "two-point probe", "hall": "Hall effect",
        "impedance": "impedance", "dc": "DC", "in_plane": "in-plane", "out_of_plane": "out-of-plane"}),
}

# Canonical names of the chemicals the model identifies in additive names (one per alias key).
CHEMICAL_NAMES = {
    "dmso": "DMSO", "eg": "ethylene glycol", "deg": "diethylene glycol", "teg": "triethylene glycol",
    "dmf": "DMF", "nmp": "NMP", "methanol": "methanol", "ethanol": "ethanol", "ipa": "isopropanol",
    "water": "water", "acetone": "acetone", "acetonitrile": "acetonitrile", "chlorobenzene": "chlorobenzene",
    "formamide": "formamide", "glycerol": "glycerol", "methoxyethanol": "2-methoxyethanol", "thf": "THF",
    "toluene": "toluene", "sulfuric_acid": "sulfuric acid", "hydrochloric_acid": "hydrochloric acid",
    "nitric_acid": "nitric acid", "phosphoric_acid": "phosphoric acid", "formic_acid": "formic acid",
    "acetic_acid": "acetic acid", "methanesulfonic_acid": "methanesulfonic acid",
    "p_toluenesulfonic_acid": "p-toluenesulfonic acid", "oxalic_acid": "oxalic acid", "sorbitol": "sorbitol",
    "gops": "GOPS", "triton_x100": "Triton X-100", "zonyl": "Zonyl", "dynol": "Dynol", "sds": "SDS",
    "sdbs": "SDBS", "peg": "PEG", "pva": "PVA", "pvp": "PVP", "pei": "PEI", "cnt": "CNT", "graphene": "graphene",
    "silver": "silver", "gold": "gold", "hydrazine": "hydrazine", "urea": "urea", "tmu": "TMU", "tdae": "TDAE",
}


def parse(field, value):
    """Categories the model's own parser finds in one field value (in the table's order)."""
    prefix, table = CATEGORY_FIELDS[field]
    science = feature_view(canonicalize({field: value}))
    if field == "polymer_grade":
        found = set(str(science.get("polymer_grade", "")).split("+"))
    else:
        found = {key[len(prefix) + 1:] for key, v in science.items()
                 if key.startswith(prefix + "_") and v == 1.0}
    return [category for category in table if category in found]


def chemicals(name):
    """Alias keys the model identifies in one additive name."""
    return list(_features._chemicals(name))


@lru_cache(maxsize=1)
def vocabulary():
    """Allowed options: {field: [canonical wordings]} plus the identified additive chemicals."""
    out = {field: list(table.values()) for field, (_, table) in CATEGORY_FIELDS.items()}
    out["additive_names"] = [CHEMICAL_NAMES[key] for key in _features.CHEMICAL_ALIASES if key in CHEMICAL_NAMES]
    return out


def canonical_categories(experiment, registered_names=()):
    """Strict mode: rewrite categorical fields and listed additive names to their canonical wording."""
    out, notes = dict(experiment), []
    for field, (_, table) in CATEGORY_FIELDS.items():
        value = out.get(field)
        if value is None:
            continue
        found = parse(field, value)
        if not found:
            raise ValueError(f"{field}={value!r} is not a recognised option; use one of: "
                             + ", ".join(table.values()))
        canonical = " + ".join(table[category] for category in found)
        if canonical != value:
            dropped = " (numbers in this text are not used; give settings in process_stages or "\
                      "extra_scalar_features)" if re.search(r"\d", str(value)) and field == "deposition_method" else ""
            notes.append(f"{field}: {value!r} -> {canonical!r}{dropped}")
        out[field] = canonical
    names = out.get("additive_names")
    if names:
        new_names = []
        for name in names:
            keys = chemicals(name)
            if name in registered_names:
                new_names.append(name)                     # an unlisted molecule already resolved from SMILES
            elif len(keys) == 1 and keys[0] in CHEMICAL_NAMES:
                canonical = CHEMICAL_NAMES[keys[0]]
                if canonical != name:
                    notes.append(f"additive_names: {name!r} -> {canonical!r}")
                new_names.append(canonical)
            elif len(keys) > 1:
                raise ValueError(f"additive name {name!r} names several chemicals ({', '.join(keys)}); "
                                 "give one chemical per entry")
            else:
                raise ValueError(f"additive name {name!r} is not an identified chemical; use one of "
                                 f"pedot_pss.vocabulary()['additive_names'] or give its SMILES in additive_smiles")
        out["additive_names"] = new_names
    return out, notes
