"""Additive chemistry from SMILES (optional; needs RDKit).

The model represents additives by identified chemical (presence and dose) and by five RDKit descriptors
of the identified molecules (molecular weight, Crippen logP, TPSA, H-bond donors and acceptors; mean,
min and max over the recipe). ``additive_smiles`` lets a caller describe an additive by structure:

* a SMILES of one of the molecules in the descriptor table is resolved to that chemical, so it predicts
  exactly like writing the chemical's name;
* an unlisted molecule is registered for the duration of the prediction with its own descriptors,
  computed with the same RDKit calls as the table. It then enters through the descriptor features only;
  the model has never seen it, so treat the prediction as an extrapolation in descriptor space.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import json
import threading

from ._vendor import features as _features
from ._categories import CHEMICAL_NAMES

_LOCK = threading.RLock()


def _rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as error:
        raise ImportError("additive_smiles needs RDKit: pip install 'pedot-pss-hybrid[chem]' "
                          "(or pip install rdkit)") from error
    RDLogger.DisableLog("rdApp.*")
    return Chem, Crippen, Descriptors, Lipinski, rdMolDescriptors


def describe(smiles):
    """Canonical SMILES, InChIKey and the table's five descriptors for one single-molecule SMILES."""
    Chem, Crippen, Descriptors, Lipinski, rdMolDescriptors = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"additive_smiles entry {smiles!r} is not a valid SMILES string")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"additive_smiles entry {smiles!r} must describe one molecule (one fragment)")
    values = {"mol_wt": float(Descriptors.MolWt(mol)),
              "mol_logp": float(Crippen.MolLogP(mol, includeHs=True, force=True)),
              "tpsa": float(rdMolDescriptors.CalcTPSA(mol, force=True, includeSandP=False)),
              "hbd": float(Lipinski.NumHDonors(mol)),
              "hba": float(Lipinski.NumHAcceptors(mol))}
    return {"canonical_smiles": Chem.MolToSmiles(mol), "inchikey": Chem.MolToInchiKey(mol), "values": values}


@lru_cache(maxsize=1)
def _listed_by_inchikey():
    entries = json.loads(_features.DESCRIPTOR_PATH.read_text(encoding="utf-8")).get("entries", {})
    return {entry["inchikey"]: key for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("status") == "verified" and entry.get("inchikey")}


def resolve_additives(experiment):
    """Remove ``additive_smiles`` and apply it to ``additive_names``.

    Returns (experiment without additive_smiles, {alias key: (pattern, descriptors)} to register, notes).
    """
    out = {key: value for key, value in experiment.items() if key != "additive_smiles"}
    smiles = experiment.get("additive_smiles")
    if smiles is None:
        return out, {}, []
    names = out.get("additive_names")
    names = list(names) if isinstance(names, list) else ([] if names is None else [names])
    if not isinstance(smiles, list) or len(smiles) != len(names):
        raise ValueError("additive_smiles must be a list aligned with additive_names; use null where no SMILES is given")
    listed = _listed_by_inchikey()
    molecules, notes = {}, []
    for i, (name, entry) in enumerate(zip(names, smiles)):
        if entry is None:
            continue
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"additive_smiles[{i}] must be a SMILES string or null")
        info = describe(entry.strip())
        hit = listed.get(info["inchikey"])
        named = list(_features._chemicals(name)) if name is not None else []
        if named:
            if hit is not None and hit not in named:
                raise ValueError(f"additive {name!r} and its SMILES {entry!r} describe different molecules")
            continue                                    # an identified name already carries the chemistry
        if hit is not None:
            names[i] = CHEMICAL_NAMES[hit]
            notes.append(f"additive {i}: SMILES {entry!r} identifies {CHEMICAL_NAMES[hit]}")
            continue
        token = "unlisted" + info["inchikey"][:14].lower()
        molecules["unlisted_" + info["inchikey"][:14].lower()] = (rf"\b{token}\b", info["values"])
        names[i] = token
        v = info["values"]
        notes.append(f"additive {name!r} ({info['canonical_smiles']}) is not in the training chemistry; it enters "
                     f"through RDKit descriptors only (MW {v['mol_wt']:.1f}, logP {v['mol_logp']:.2f}, "
                     f"TPSA {v['tpsa']:.1f}, HBD {v['hbd']:.0f}, HBA {v['hba']:.0f}); treat as an extrapolation")
    out["additive_names"] = names
    return out, molecules, notes


@contextmanager
def registered(molecules):
    """Temporarily add unlisted molecules to the feature builder's chemical aliases and descriptor registry."""
    if not molecules:
        yield
        return
    with _LOCK:
        original = _features._molecular_registry
        registry = dict(original())
        added = []
        try:
            for key, (pattern, values) in molecules.items():
                if key not in _features.CHEMICAL_ALIASES:
                    _features.CHEMICAL_ALIASES[key] = (pattern,)
                    added.append(key)
                registry[key] = dict(values)
            _features._molecular_registry = lambda: registry
            yield
        finally:
            for key in added:
                _features.CHEMICAL_ALIASES.pop(key, None)
            _features._molecular_registry = original
