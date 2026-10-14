"""Screen candidate additives with the surrogate and report the throughput.

Every molecule in candidate_molecules.csv (232 molecules outside the training data, given by SMILES) is added to the
planned spin-coating recipe at several doses; the batch is predicted with one call and ranked. The script then uses
the measured throughput to estimate how long brute-force enumeration of larger search spaces would take.

    python -m pip install '.[chem]'          # RDKit, needed for molecules given by SMILES
    python examples/screen_candidates.py
"""
import csv
import json
import time
from pathlib import Path

import pedot_pss

HERE = Path(__file__).resolve().parent
DOSES_VOL_PCT = (1, 3, 5, 10)

base = json.loads((HERE / "planned_spin_coating.json").read_text())
base["dose_basis"] = "additive volume per volume of PEDOT:PSS stock dispersion"
# Rewrite the base recipe to canonical wording once; the batch can then skip the per-recipe synonym rewrite
# (canonical=False), which is about 4x faster and gives identical predictions for canonical inputs.
base = pedot_pss.normalize(base)
candidates = list(csv.DictReader((HERE / "candidate_molecules.csv").open()))
batch = [dict(base, additive_names=[c["name"]], additive_amounts_raw=[f"{dose} vol%"], additive_smiles=[c["smiles"]])
         for c in candidates for dose in DOSES_VOL_PCT]

pedot_pss.load_model()                                   # load once (a few seconds); excluded from the timing
start = time.perf_counter()
results = pedot_pss.predict_many(batch, canonical=False)
seconds = time.perf_counter() - start
rate = len(batch) / seconds

ranked = sorted(zip(batch, results), key=lambda pair: -pair[1]["conductivity_S_cm"])
print(f"{len(batch)} recipes ({len(candidates)} molecules x {len(DOSES_VOL_PCT)} doses) in {seconds:.1f} s "
      f"= {rate:.0f} recipes/s in one process\n")
print("Top 10 candidates")
for recipe, result in ranked[:10]:
    print(f"  {result['conductivity_S_cm']:8.1f} S/cm  {recipe['additive_names'][0]} at {recipe['additive_amounts_raw'][0]}")

print("\nBrute-force enumeration at this rate (one process; scale by the number of processes):")
year = 365.25 * 24 * 3600
for label, size in (("1e6 recipes", 1e6), ("1e9 recipes", 1e9),
                    ("1e17 recipes (one additive from 1e4 molecules, coarse grids)", 1e17),
                    ("1e21 recipes (additive drawn from PubChem-scale chemistry)", 1e21)):
    t = size / rate
    print(f"  {label:62s} {t / 3600:12.3g} h  ({t / year:.3g} years)")
