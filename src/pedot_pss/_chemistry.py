"""Molecules given by structure (SMILES) for additives and post-treatment chemicals.

The model describes every additive and post-treatment chemical by cheminformatics descriptors (35 RDKit
descriptors and functional-group counts, structure classes such as acid / polar solvent / ionic liquid /
surfactant, and the dose), never by its name. Listed molecules come with the bundled chemistry table
(``_vendor/chem_table.json``, 329 additive names and 110 post-treatment chemicals). Any other molecule can be
given by SMILES (needs RDKit: ``pip install 'pedot-pss-hybrid[chem]'``):

* ``additive_smiles``: a list aligned with ``additive_names`` (a SMILES or null per additive);
* ``post_treatment_smiles``: ``{name as written in post_treatment: SMILES}``; concentration and role (active
  treatment or rinse) are read from the ``post_treatment`` text exactly as for listed chemicals.

The descriptors of a SMILES are computed with the same RDKit calls as the bundled table. A SMILES of a listed
molecule (same InChIKey) is resolved to that entry. New molecules are registered only for the duration of the
prediction.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import re
import threading

from ._vendor import openrep as _openrep

_LOCK = threading.RLock()


def _rdkit():
    try:
        from ._vendor import chem_descriptors
        from rdkit import Chem
    except ImportError as error:
        raise ImportError("Molecules given by SMILES need RDKit: pip install 'pedot-pss-hybrid[chem]' "
                          "(or pip install rdkit)") from error
    return chem_descriptors, Chem


def describe(smiles):
    """Table-style record (structure, RDKit descriptors, InChIKey) for one SMILES string."""
    cd, Chem = _rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("A SMILES must be a nonempty string")
    try:
        mol, _, _ = cd.prepare(smiles.strip())
    except Exception as error:
        raise ValueError(f"{smiles!r} is not a valid SMILES string") from error
    if mol is None:
        raise ValueError(f"{smiles!r} is not a valid SMILES string")
    form = Chem.MolToSmiles(mol)
    inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles.strip())) or Chem.MolToInchiKey(mol)
    fragments = len(Chem.GetMolFrags(mol))
    return {"smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles.strip())), "smiles_descriptor_form": form,
            "class": "salt_or_ionic_liquid" if fragments > 1 else "small_molecule", "rdkit": cd.rd(mol),
            "inchikey_rdkit": inchikey, "group_key": "inchikey:" + inchikey}


@lru_cache(maxsize=1)
def _listed():
    """InChIKey -> listed additive name (the spelling most frequent in the training data) / post-treatment key."""
    rep = _openrep.chem()
    names = {}
    for name, rec in sorted(rep.names.items(), key=lambda kv: (-(kv[1].get("mentions") or 0), kv[0])):
        if rec.get("inchikey_rdkit"):
            names.setdefault(rec["inchikey_rdkit"], name)
    post = {}
    for key, rec in sorted(rep.post.items()):
        if rec.get("inchikey_rdkit"):
            post.setdefault(rec["inchikey_rdkit"], key)
    return names, post


def listed_additive_names():
    return sorted(_openrep.chem().names)


def listed_post_treatment_chemicals():
    rep = _openrep.chem()
    return sorted({rec.get("canonical_name") or key for key, rec in rep.post.items()})


def resolve_structures(experiment):
    """Remove ``additive_smiles`` / ``post_treatment_smiles`` and resolve them.

    Returns (experiment without the two fields, {"names": {name: record}, "post": {key: record}}, notes).
    """
    out = {k: v for k, v in experiment.items() if k not in ("additive_smiles", "post_treatment_smiles")}
    overrides, notes = {"names": {}, "post": {}}, []
    smiles = experiment.get("additive_smiles")
    if smiles is not None:
        names = out.get("additive_names")
        names = list(names) if isinstance(names, list) else ([] if names is None else [names])
        if not isinstance(smiles, list) or len(smiles) != len(names):
            raise ValueError("additive_smiles must be a list aligned with additive_names; use null where no SMILES "
                             "is given")
        rep = _openrep.chem()
        listed_names, _ = _listed()
        for i, (name, entry) in enumerate(zip(names, smiles)):
            if entry is None:
                continue
            rec = describe(entry)
            hit = listed_names.get(rec["inchikey_rdkit"])
            known = rep.names.get(name) if isinstance(name, str) else None
            if known is not None and known.get("inchikey_rdkit") and known["inchikey_rdkit"] != rec["inchikey_rdkit"]:
                raise ValueError(f"additive {name!r} and its SMILES {entry!r} describe different molecules")
            if hit is not None:
                if hit != name:
                    notes.append(f"additive {i}: SMILES {entry!r} is the listed molecule {hit!r}")
                names[i] = hit
                continue
            label = name if isinstance(name, str) and name.strip() else f"molecule {i}"
            if label in rep.names:
                raise ValueError(f"additive name {label!r} is a listed molecule with a different structure")
            overrides["names"][label] = rec
            names[i] = label
            notes.append(f"additive {label!r} enters by its structure ({rec['smiles']}); it is not in the training "
                         "chemistry, so treat the prediction as an extrapolation in descriptor space")
        out["additive_names"] = names
    post = experiment.get("post_treatment_smiles")
    if post is not None:
        if not isinstance(post, dict) or not all(isinstance(k, str) and k.strip() for k in post):
            raise ValueError("post_treatment_smiles must map a chemical name used in post_treatment to its SMILES")
        text = str(out.get("post_treatment") or "").lower()
        _, listed_post = _listed()
        rep = _openrep.chem()
        for name, entry in post.items():
            if name.strip().lower() not in text:
                raise ValueError(f"post_treatment_smiles name {name!r} does not appear in post_treatment")
            rec = describe(entry)
            hit = listed_post.get(rec["inchikey_rdkit"])
            if hit is not None:
                rec = dict(rep.post[hit])
                notes.append(f"post-treatment {name!r}: SMILES is the listed chemical {hit!r}")
            else:
                notes.append(f"post-treatment chemical {name!r} enters by its structure ({rec['smiles']}); it is not "
                             "in the training chemistry, so treat the prediction as an extrapolation")
            rec["pattern"] = r"(?<![\w])" + re.escape(name.strip().lower()) + r"(?![\w])"
            rec.pop("negation_pattern", None)
            overrides["post"]["user:" + name.strip().lower()] = rec
    return out, overrides, notes


def resolve_listed_aliases(experiment, overrides):
    """Additive names outside the additive table that name exactly one identified chemical of the post-treatment
    table (e.g. 'chlorobenzene') get that chemical's structure. Training additive names are all in the additive
    table, so this only widens the inputs that are accepted."""
    from ._vendor import features as _features
    rep = _openrep.chem()
    for name in experiment.get("additive_names") or []:
        if not isinstance(name, str) or name in rep.names or name in overrides["names"]:
            continue
        keys = list(_features._chemicals(name))
        if len(keys) == 1 and keys[0] in rep.post:
            overrides["names"][name] = rep.post[keys[0]]
    return overrides


def merge(target, overrides):
    """Merge one experiment's overrides into a batch-wide set (same name -> same structure required)."""
    for kind in ("names", "post"):
        for key, rec in overrides[kind].items():
            have = target[kind].get(key)
            if have is not None and have.get("inchikey_rdkit") != rec.get("inchikey_rdkit"):
                raise ValueError(f"{key!r} is given with two different structures in one batch")
            target[kind][key] = rec
    return target


@contextmanager
def registered(overrides):
    """Temporarily add molecules given by SMILES to the bundled chemistry adapter."""
    if not overrides or not (overrides.get("names") or overrides.get("post")):
        yield
        return
    with _LOCK:
        rep = _openrep.chem()
        added_names, added_post, added_patterns = [], [], []
        try:
            for name, rec in overrides["names"].items():
                if name not in rep.names:
                    rep.names[name] = rec
                    added_names.append(name)
            for key, rec in overrides["post"].items():
                if key not in rep.post:
                    rep.post[key] = rec
                    added_post.append(key)
                    pattern = (key, re.compile(rec["pattern"]))
                    rep._post_patterns.append(pattern)
                    added_patterns.append(pattern)
            yield
        finally:
            for name in added_names:
                rec = rep.names.pop(name, None)
                rep._entity.pop(id(rec), None)
            for key in added_post:
                rec = rep.post.pop(key, None)
                rep._entity.pop(id(rec), None)
            for pattern in added_patterns:
                rep._post_patterns.remove(pattern)
