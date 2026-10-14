"""RDKit descriptors of a molecule given by SMILES: the m-chem v1.3 table calls (build_table.rd/prepare).

Needs RDKit (pip install 'pedot-pss-hybrid[chem]'). Imported only for molecules given by SMILES.
"""
from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

CORE = ["mol_wt", "mol_logp", "tpsa", "hbd", "hba"]
EXT = ["num_rotatable_bonds", "fraction_csp3", "num_aromatic_rings", "heavy_atom_count", "mol_mr",
       "formal_charge", "fragment_count", "positive_charge_count", "negative_charge_count"]
FG = {  # SMARTS functional-group counts (unique matches)
    "sulfonic_acid_or_sulfonate": "[#6][SX4](=O)(=O)[OX2H1,OX1-]",
    "sulfate_group": "[OX2,OX1-;!$(O[#6]);!$(O[SX4](=O)(=O)[#6])][SX4](=O)(=O)[OX2,OX1-]",
    "sulfate_ester": "[#6][OX2][SX4](=O)(=O)[OX2H1,OX1-,OX2]",
    "sulfonyl_imide": "[SX4](=O)(=O)[NX2-,NX3H1][SX4](=O)=O",
    "sulfoxide": "[#6][SX3](=O)[#6]",
    "sulfone": "[#6][SX4](=O)(=O)[#6]",
    "hydroxyl": "[OX2H1][#6;!$([#6]=[O,S,N])]",
    "carboxylic_acid_or_carboxylate": "[CX3](=O)[OX2H1,OX1-]",
    "amide": "[NX3][CX3](=[OX1])",
    "amine": "[NX3;!$(N[#6,#16,#15]=[O,S,N]);!$(N-[SX4]);!$(N#*);!$(N=*);!$([N+]);!$(N-N=*)]",
    "ether": "[OD2;!$(O[#6]=O);!$(O[S,P]);!$(O-O)]([#6])[#6]",
    "ester": "[#6][CX3](=O)[OX2][#6]",
    "epoxide": "C1OC1",
    "nitrile": "[CX2]#[NX1]",
    "halogen_covalent": "[F,Cl,Br,I;X1;+0]-*",
    "halide_ion": "[F-,Cl-,Br-,I-]",
    "imidazolium": "[n+]1cncc1",
    "pyridinium": "[n+;r6]",
    "ammonium": "[NX4+]",
    "phosphonium": "[PX4+]",
    "phosphate": "[PX4](=O)([OX2,OX1-])([OX2,OX1-])[OX2,OX1-]",
}
FG_PATT = {k: Chem.MolFromSmarts(v) for k, v in FG.items()}
assert all(FG_PATT.values()), [k for k, v in FG_PATT.items() if v is None]
DESCRIPTOR_NAMES = CORE + EXT + ["fg_" + k for k in FG]
METALS = {3, 11, 19, 37, 55, 4, 12, 20, 38, 56, 13, 31, 49, 81, 50, 82, 83} | set(range(21, 31)) | \
    set(range(39, 49)) | set(range(72, 81)) | set(range(57, 72))



# ----------------------------------------------------------------------------
def rd(mol) -> dict:
    out = {
        "mol_wt": Descriptors.MolWt(mol),
        "mol_logp": Crippen.MolLogP(mol, includeHs=True, force=True),
        "tpsa": rdMolDescriptors.CalcTPSA(mol, force=True, includeSandP=False),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "num_rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)) if any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()) else 0.0,
        "num_aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "heavy_atom_count": float(mol.GetNumHeavyAtoms()),
        "mol_mr": Crippen.MolMR(mol),
        "formal_charge": float(Chem.GetFormalCharge(mol)),
        "fragment_count": float(len(Chem.GetMolFrags(mol))),
        "positive_charge_count": float(sum(max(a.GetFormalCharge(), 0) for a in mol.GetAtoms())),
        "negative_charge_count": float(sum(-min(a.GetFormalCharge(), 0) for a in mol.GetAtoms())),
    }
    for k, patt in FG_PATT.items():
        out["fg_" + k] = float(len(mol.GetSubstructMatches(patt, uniquify=True)))
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def has_metal_bond(mol) -> bool:
    return any(a.GetAtomicNum() in METALS and a.GetDegree() > 0 for a in mol.GetAtoms())


def prepare(smiles: str):
    """Parse; disconnect covalently drawn metals; drop hydrate waters. Returns (mol_for_descriptors, notes, n_waters)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES {smiles}")
    notes = []
    if has_metal_bond(mol):
        mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
        Chem.SanitizeMol(mol)
        if has_metal_bond(mol):  # e.g. In-Cl is not in the disconnector's default list: break remaining bonds
            rw = Chem.RWMol(mol)
            for b in list(rw.GetBonds()):
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                metal, other = (a1, a2) if a1.GetAtomicNum() in METALS else (a2, a1)
                if metal.GetAtomicNum() in METALS and other.GetAtomicNum() not in METALS and b.GetBondType() == Chem.BondType.SINGLE:
                    rw.RemoveBond(a1.GetIdx(), a2.GetIdx())
                    metal.SetFormalCharge(metal.GetFormalCharge() + 1)
                    other.SetFormalCharge(other.GetFormalCharge() - 1)
                    metal.SetNoImplicit(True)
            mol = rw.GetMol()
            Chem.SanitizeMol(mol)
        notes.append("metal-ligand bonds disconnected into ions (rdMolStandardize.MetalDisconnector)")
    frags = Chem.GetMolFrags(mol, asMols=True)
    waters = [f for f in frags if Chem.MolToSmiles(f) == "O"]
    if waters and len(frags) > len(waters):
        keep = [f for f in frags if Chem.MolToSmiles(f) != "O"]
        mol = Chem.MolFromSmiles(".".join(Chem.MolToSmiles(f) for f in keep))
        notes.append(f"{len(waters)} water(s) of crystallisation removed")
    return mol, notes, len(waters) if waters and len(frags) > len(waters) else 0


