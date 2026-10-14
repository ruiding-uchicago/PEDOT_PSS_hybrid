# PEDOT:PSS experimental conductivity surrogate

Give the tool a planned PEDOT:PSS experiment; it returns predicted conductivity in **S/cm** and **log10(S/cm)**.
You supply material and processing conditions; the tool builds its own descriptors, physical coordinates and
intermediate predictions. It is a physics + ML hybrid (general effective medium transport, an interface circuit,
dedicated acid/salt/PEO/filler branches and thermal-history clocks, corrected by data-driven learners).

**Every chemical enters as cheminformatics descriptors.** Additives and post-treatment chemicals are described by
their molecular structure (35 RDKit descriptors and functional-group counts, structure classes such as acid, polar
solvent, ionic liquid or surfactant, and the dose), grade and substrate by nominal material properties. No name
one-hot, text embedding or bag-of-words feature is used. Any molecule with a structure can be evaluated: 329
additives and 110 post-treatment chemicals are listed, and any other molecule can be given by SMILES. This makes the
input space far larger than a grid search can cover, which is what the surrogate is for.

Complete fitted model; standalone inference verified in a fresh environment (macOS arm64, Python 3.12). Linux uses
the same code path but has not been tested yet. The fitted model is the asset of GitHub Release v0.2.0.
The tool returns the physics-preferred output **P**; **M** is the companion benchmark.

## Five-fold results

Data cleaning: 536 low-quality literature points (8% of the literature rows) were removed with the `keep_v2_cap08`
rule (consensus of out-of-fold residuals from five model families plus independent extraction evidence: non-PEDOT:PSS
controls, quoted values, inconsistent duplicates, stretch/bend series, aggregates, value flags). Removed points are
excluded from training and scoring; the 75 Polybot runs are never removed. Pooled held-out predictions on fixed
random-row five folds, 6,244 points (6,169 from 429 papers + 75 Polybot runs):

| Output | All 6,244 R² | Literature 6,169 R² | All RMSE (dex) |
|---|---:|---:|---:|
| P (default) | 0.960 | 0.961 | 0.365 |
| M | 0.960 | 0.961 | 0.364 |

The final model reproduces the 75 Polybot runs it was trained on with R² 0.962; that is a training fit, not a
held-out score.

How to read these numbers. The cleaning rule used out-of-fold residuals, so report them as "after data cleaning".
R² and RMSE use log10(S/cm). Random-row folds measure prediction within this corpus. For chemicals the model has
never seen, a chemical-grouped evaluation (all rows with a given additive held out together) gives R² 0.48 on
those rows with the descriptor representation, against 0.42 for the same model with name identities. See the
aggregate [evaluation](src/pedot_pss/models/evaluation.json) and the [model card](MODEL_CARD.md).

## Use

Use Python 3.12 in an independent virtual environment (macOS/Linux):

```bash
git clone https://github.com/ruiding-uchicago/PEDOT_PSS_hybrid.git
cd PEDOT_PSS_hybrid
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[chem]'        # [chem] adds RDKit, needed only for molecules given by SMILES
python -m pedot_pss setup-model
python -m pedot_pss examples/polybot_recipe.json
```

`setup-model` downloads the complete fitted model from the pinned GitHub Release (it is excluded from Git).
`PEDOT_PSS_MODEL` or `--model PATH` selects a locally installed model file.

```python
import json
from pathlib import Path
import pedot_pss

experiment = json.loads(Path("examples/planned_spin_coating.json").read_text())
print(pedot_pss.predict(experiment)["conductivity_S_cm"])

# a molecule that is not in the training data, given by structure
candidate = dict(experiment, additive_names=["ethyl methyl sulfone"], additive_smiles=["CCS(C)(=O)=O"])
print(pedot_pss.predict(candidate))

# the loaded model is reused within the process; prefer batches in loops
predictions = pedot_pss.predict_many([experiment, candidate])
```

The command-line interface accepts a dictionary or a list of dictionaries:

```bash
pedot-pss experiments.json --output predictions.json
```

Prediction never fits a model and never transmits inputs. `PEDOT_PSS_PREDICT_THREADS` (default 1) sets the threads
of the fitted tree learners; one thread gives bitwise-repeatable predictions.

## How the inputs become numbers

Every recipe becomes one numerical vector (about 950 features). Text fields reach the model only through
rule-based parsing:

| Input | Encoding |
|---|---|
| `additive_names` + `additive_amounts_raw` (+ optional `additive_smiles`) | Per molecule: 35 RDKit descriptors (molecular weight, Crippen logP, TPSA, H-bond donors/acceptors, rotatable bonds, sp3 fraction, aromatic rings, heavy atoms, molar refractivity, charges, fragments, and 21 functional-group counts such as sulfonic acid, sulfate, sulfoxide, sulfone, hydroxyl, carboxylic acid, amide, amine, ether, imidazolium, ammonium) and structure classes (acid, polar solvent, ionic liquid, metal salt, surfactant, base, polymer, nanocarbon, nanometal). The dose becomes a saturation d/(d + h). Per recipe: dose-weighted means, minimum/maximum of size, logP and TPSA, and the full descriptors of the two highest-dose molecules. |
| `post_treatment` (+ optional `post_treatment_smiles`) | Chemicals are detected in the text (110 listed) or given by SMILES and described the same way; the concentration is read from the text; chemicals in rinse/wash clauses are kept apart from the active treatment. |
| `polymer_grade` | Material descriptors: PSS:PEDOT ratio, solids content, conductivity class, paste / re-dispersible / in-house flags (nominal supplier datasheet values). |
| `substrate` | Material descriptors: water contact angle, surface energy, thermal conductivity, conductive, flexible, porous, polymer/inorganic, transparent, maximum processing temperature. |
| `deposition_method`, `specimen_geometry`, `measurement_method` | Fixed categories (spin coating, blade coating, drop casting, ...; film, fiber, printed trace, ...; four-point probe, two-point probe, Hall effect, ...). |
| Temperatures, times, solids, ratios, `process_stages`, `extra_scalar_features` | Numbers in fixed units, each with a "given" indicator (unknown is not zero). |

The physics layers additionally use a few recognized-chemical coordinates (DMSO/EG volume fraction, acid or
polar-solvent post-treatment, four common grades) and the dedicated acid/salt/PEO/SWCNT/DAE branches; a new molecule
enters the data-driven layers through its descriptors and triggers these physics terms only if it is one of the
recognized chemicals.

### Fixed options, synonyms and strict mode

By default `predict`, `predict_many` and `normalize` rewrite `polymer_grade`, `deposition_method`, `substrate`,
`specimen_geometry`, `measurement_method` and every additive name to one canonical wording per option, with the
same function (`pedot_pss._canonical`) that rewrote the training data. Synonyms therefore give identical
predictions: "ethylene glycol", "ethylene glycol (EG)" and "EG" are one option, "Mylar" is PET, "Clevios PH1000"
is PH1000. A rewrite is made only if the model's parsers read exactly the same information from the new wording,
and the result's `"notes"` list every rewrite.

`pedot_pss.vocabulary()` returns the fixed categories, the 329 listed additive names and the 110 listed
post-treatment chemicals. A SMILES of a listed molecule (for example `CS(C)=O`) resolves to that molecule and
predicts exactly as its name. An unlisted name without a SMILES has no structure and is noted; `strict=True`
rejects it, together with categorical options outside the vocabulary. `canonical=False` uses all text as written.

### Molecules by structure (SMILES)

`additive_smiles` is an optional list aligned with `additive_names` (a SMILES or `null` per additive);
`post_treatment_smiles` maps a chemical name used in `post_treatment` to its SMILES:

```python
pedot_pss.predict(dict(experiment,
    post_treatment="immersed in 1 M Xacid for 10 min; rinsed with DI water",
    post_treatment_smiles={"Xacid": "OS(=O)(=O)C(F)(F)F"}))
```

Descriptors are computed with the same RDKit calls as the bundled table (`pip install '.[chem]'`). Molecules that
are not in the training chemistry are an extrapolation in descriptor space and the result's `"notes"` say so.
In the spin-coating example (5 vol% additive), 232 molecules outside the training data receive 227 distinct predictions between 0.3 and 621 S/cm (DMSO 559 S/cm, no additive 0.8 S/cm); glycols and polyols rank highest (median 256 S/cm), followed by sulfonic acids and polar aprotic solvents (about 190 S/cm), and sugars and sugar alcohols, carboxylic and other weak acids and simple salts rank lowest (24-77 S/cm).
Use such predictions to rank candidates, and confirm with experiments.

## For agent developers

The package is meant to be called as a black-box tool inside optimization or agent loops.

- **Call:** `pedot_pss.predict(experiment)` takes one experiment dictionary and returns
  `{"conductivity_S_cm": float, "log10_conductivity_S_cm": float}` (plus `"notes"` when an input was rewritten,
  is unlisted, or was given by structure). `pedot_pss.predict_many(list_of_experiments)` returns a list in order.
- **Tool schema:** [`input_schema.json`](input_schema.json) is a JSON Schema for the experiment dictionary and can
  be used directly as the parameter schema of an LLM tool/function definition.
- **Search space:** vary the numbers (doses, temperatures, times, speeds), the additives (any listed molecule or any
  SMILES) and the post-treatment chemical; `pedot_pss.vocabulary()` lists the fixed options.
- **Speed (laptop CPU):** loading the model takes a few seconds (the very first import after installation takes
  about 30 s while Python compiles the package); a batch of 19 recipes then takes about 2 s. No GPU and no
  PyTorch are needed.
- **Deterministic:** the same input always returns the same prediction.
- **Errors:** unknown field names, outcome or provenance fields (DOI, paper IDs, measured conductivity, SD,
  coverage, ...), malformed numbers, units or SMILES, and unsupported process-stage kinds raise
  `ValueError`/`TypeError` with a message naming the offending field. Use `null` or omit a field when a condition is
  unknown; do not use 0.
- **Scope:** predictions are most reliable inside the formulations and processes covered by the training data
  (literature PEDOT:PSS films plus the Polybot blade-coating platform). Treat predictions far outside it as guidance
  for which experiment to run next, and update with real measurements.

```python
import json
from pathlib import Path
import pedot_pss

SCHEMA = json.loads(Path("input_schema.json").read_text())   # parameters for your tool definition

def predict_conductivity(experiment: dict) -> dict:
    """Tool: planned PEDOT:PSS experiment in, predicted conductivity (S/cm) out."""
    try:
        return pedot_pss.predict(experiment)
    except (TypeError, ValueError) as error:
        return {"error": str(error)}

print(predict_conductivity(json.loads(Path("examples/planned_spin_coating.json").read_text())))
```

## Experimental input

There is **one dictionary format with 24 optional fields**, plus the two optional structure fields
`additive_smiles` and `post_treatment_smiles`. Values include numbers, material/process descriptions, additive lists
and ordered process stages.

| Type | Fields |
|---|---|
| Material and amount descriptions | `polymer_grade`, `stock_solids_scope`, `pedot_pss_ratio_raw`, `dose_basis` |
| Process descriptions | `deposition_method`, `substrate`, `specimen_geometry`, `post_treatment`, `process_sequence` |
| Planned measurement descriptions | `measurement_method`, `measurement_direction`, `measurement_quantity` |
| Numeric controls | `stock_solids_wt_percent`, `dry_temperature_C`, `dry_duration_min`, `anneal_temperature_C`, `anneal_duration_min`, `post_treatment_temperature_C`, `post_treatment_duration_min`, `measurement_temperature_C` |
| Lists | `additive_names`, `additive_amounts_raw`, `extra_scalar_features`, `process_stages` |
| Optional structures | `additive_smiles`, `post_treatment_smiles` |

Supply known conditions. Omitted fields or `null` mean unknown; known zero remains zero. At least one real planned
condition is required.

Scalar temperatures use °C and scalar durations use minutes. An extra scalar is
`{ "name": ..., "value": ..., "unit": ... }`, chosen from the 56 allowed experimental controls. Structured
`process_stages` preserve order; their durations use seconds and temperatures use °C. Unit conversion at the public
boundary applies to the following **extra scalars** only:

| Extra scalar | Normalized unit |
|---|---|
| `planned_via_diameter`, `blade_gap` | um |
| `measurement_bias` | V |
| `blend_pedot_pss_dry_mass_percent` | % |
| The six `preparation_water_*_mg_l` controls | mg/L |
| `preparation_water_ph` | dimensionless |
| `stock_volume` | mL |
| `mix_duration`, `substrate_uvo_duration`, `post_dry_duration`, `second_anneal_duration`, `air_storage_time` | s |
| `mix_rotation_speed` | rpm |
| Extra scalar names containing `temperature` | C |

Numeric controls require finite JSON numbers. Numeric strings and booleans are rejected, durations cannot be
negative, temperatures must exceed −273.15 °C, and stock solids must be between 0 and 100 wt%. Additive names and
amounts use equal-length lists; use `null` for an unknown amount. Amount strings keep the concentration and basis,
for example `"4 vol%"` or `"0.1 M"`.

See [input_schema.json](input_schema.json) and the example recipes in [examples/](examples/).
`pedot_pss.normalize(experiment)` displays the normalized 24-field dictionary.

DOI, paper/source/series/sample IDs, measured conductivity or its SD, measured coverage and characterization of the
finished film are excluded. Unknown top-level fields are rejected. Do not place identifiers or outcomes inside
experimental description strings; the input version `experiment-input-v3-text-clean-1012` removes identifier,
outcome and literature-extraction clauses from experimental text before parsing.

## Output

```json
{
  "conductivity_S_cm": 12.3,
  "log10_conductivity_S_cm": 1.0899051114393979
}
```

These numbers illustrate the output format; they are not a model evaluation.

## Model structure

- **Global base:** ten GEM + residual-learner candidates (LightGBM, CatBoost and ExtraTrees variants on feature
  families built from the encoded recipe), a calibrated GEM physics reference, 17 meta models and a Ridge fusion.
  During training the residual learners also see copies of the rows with the chemical identity coordinates masked,
  so the descriptor path is learned everywhere.
- **Physics:** two-phase McLachlan general effective medium transport with a positive interface resistance,
  anisotropy and bounded bypass; acid, salt, PEO, SWCNT and pure-DAE branches activate only where the experiment
  defines their protocol and only inside the range covered by training (otherwise the parent prediction is kept).
- **Late layer:** a global recipe kernel (Gaussian anchors, Ridge) on the encoded vector is fitted as a possible late
  correction; its non-negative weight learned on the full training data is 0 (0 to 0.32 across the five folds), so the
  released model is the base + physics + P chain without a late correction.
- **Physics-preferred P:** two cumulative Arrhenius clocks (thermal ordering and damage) re-solve the effective
  medium and the interface circuit from the recipe's thermal history.
- **Polybot platform branch** (PH1000, blade coating on silicon, film, all nine planned controls): a two-state
  physical model, a percolating-film state (process GEM plus a bounded SVR residual) and a dewetted state, mixed by a
  DMSO-gated failure probability from two Gaussian-process classifiers on film-formation coordinates (blade-coating
  regime number from capillary number and Antoine water evaporation flux; solvent-wash contact time and
  temperature).

All learned parameters are fitted inside the training data of each fold. The package contains model code, the
chemistry table (PubChem structures and RDKit descriptors), the material table, input/output examples and the
pinned fitted model artifact. It contains no data corpus or private workflow files; prediction needs no private
project directory. See [MODEL_CARD.md](MODEL_CARD.md).

## Third-party notices

Chemical structures were resolved with PubChem (public domain data, NCBI); descriptors are computed with RDKit (BSD).
The optional `setup-encoder` command, kept only for older model files, fetches the MatBERT encoder of the Lawrence
Berkeley National Laboratory NLP project (LBNLP); the v0.2.0 model contains no MatBERT component. The original MIT
notice is preserved in [third_party/MatBERT_LICENSE.txt](third_party/MatBERT_LICENSE.txt).
