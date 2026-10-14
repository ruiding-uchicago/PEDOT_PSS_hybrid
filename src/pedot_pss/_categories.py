"""Fixed-option wording for the categorical fields; listed chemistry for additive names.

The model reads the categorical text fields (grade, deposition method, substrate, specimen geometry, measurement
method) only through its rule-based parser, as fixed categories (one-hot flags); grade and substrate additionally
map to nominal material descriptors. Additives and post-treatment chemicals enter as cheminformatics descriptors
of their structures (see pedot_pss._chemistry). By default every prediction first rewrites the categorical fields
and additive names to one canonical wording per option (pedot_pss._canonical, the rewrite the training corpus
received), so synonyms give identical predictions. ``strict`` mode also rejects categorical options outside
vocabulary() and additives without a structure (not listed and no SMILES).
"""
from __future__ import annotations

from functools import lru_cache

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
    """Listed options. Categorical fields: {field: [canonical wordings]}. additive_names: the molecules of the
    bundled chemistry table (any other molecule can be given by SMILES); post_treatment_chemicals: the chemicals
    detected in post_treatment text (others via post_treatment_smiles)."""
    from ._chemistry import listed_additive_names, listed_post_treatment_chemicals
    out = {field: list(table.values()) for field, (_, table) in CATEGORY_FIELDS.items()}
    out["additive_names"] = listed_additive_names()
    out["post_treatment_chemicals"] = listed_post_treatment_chemicals()
    return out


def unlisted_options(experiment, registered_names=()):
    """[(field, value)] for categorical values with no listed category and additive names without a structure
    (not in the chemistry table and not given by SMILES, ``registered_names``)."""
    from ._vendor import openrep
    out = []
    for field in CATEGORY_FIELDS:
        value = experiment.get(field)
        if isinstance(value, str) and not parse(field, value):
            out.append((field, value))
    names = openrep.chem().names
    for name in experiment.get("additive_names") or []:
        if isinstance(name, str) and name not in names and name not in registered_names:
            out.append(("additive_names", name))
    return out
