# PEDOT:PSS experimental conductivity surrogate

Give the tool a planned PEDOT:PSS experiment; it returns predicted conductivity
in **S/cm** and **log10(S/cm)**. You supply material and processing conditions.
The tool builds its own numerical features and intermediate predictions.

**Complete fitted experiment-only model; standalone inference verified in a fresh
environment (macOS arm64, Python 3.12).** Linux uses the same code path but has not
been tested yet. The GitHub model asset has not been published yet; the setup
commands below describe the planned release.
The tool returns physics-preferred **P**. M is the companion benchmark.
The fitted model is supplied as one GitHub Release asset; the repository contains
the interface, calculation code, input/output examples and aggregate evaluation.

## Actual fixed five-fold results

Data cleaning: 536 low-quality literature points (8% of the literature rows) were removed with the
`keep_v2_cap08` rule (consensus of out-of-fold residuals from five model families plus independent
extraction evidence: non-PEDOT:PSS controls, quoted values, inconsistent duplicates, stretch/bend
series, aggregates, value flags). Removed points are excluded from training and from scoring; the 75
Polybot runs are never removed. Pooled held-out predictions on the original fixed random-row five folds,
6,244 points:

| Output | All 6,244 R² | Literature 6,169 R² | All RMSE (dex) |
|---|---:|---:|---:|
| P (default) | 0.960 | 0.961 | 0.364 |
| M | 0.960 | 0.961 | 0.363 |

The final model reproduces the 75 Polybot runs it was trained on with R² 0.962; that is a training
fit, not a held-out score.

How to read these numbers. The cleaning rule used out-of-fold residuals, so report them as "after
data cleaning". The late-head combiner and physics guards were selected on these folds; choosing
the combiner, guards and P cap leave-one-fold-out gives P 0.960 (0.95974) / M 0.960 (0.95992).
R² and RMSE use log10(S/cm). These folds measure interpolation on
this corpus, not new-laboratory or control-range extrapolation. See the aggregate
[evaluation](src/pedot_pss/models/evaluation.json).

## Use

Use Python 3.12 in
an independent virtual environment (macOS/Linux):

```bash
git clone https://github.com/ruiding-uchicago/PEDOT_PSS_hybrid.git
cd PEDOT_PSS_hybrid
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m pedot_pss setup-model
python -m pedot_pss examples/polybot_recipe.json
```

`setup-model` downloads the complete fitted surrogate from the pinned GitHub
Release. It is excluded from Git. `PEDOT_PSS_MODEL` or `--model PATH` can select
a locally installed model file.


### Text fields and the text encoder

Free-text fields are first parsed into categories and numbers (grade, deposition method, substrate,
specimen geometry, measurement method, additive identities with doses, process-stage temperatures and
times). These parsed features drive the physics and most of the model. The raw text also enters two small
residual-correction heads (a word TF-IDF head and a text-plus-numeric LightGBM head). The frozen MatBERT
head of the original model family receives combiner weight 0 in this model, so prediction never runs the
MatBERT encoder: no PyTorch install, no 438 MB download and no encoder process are needed. (`setup-encoder`
and the optional `[encoder]` dependencies remain only for models that weight that head.)

Use the canonical wording below. In our checks, rephrasing the same condition changed a prediction by at
most 0.05 log10 units (about 12%). Recognised categories, with training-set counts:

| Field | Recognised categories |
|---|---|
| `polymer_grade` | PH1000 (3,890), Baytron P (494), AI4083 (116), PH500 (76), SV3 (7) |
| `deposition_method` | spin coating (3,572), drop casting (892), casting (702), wet spinning (315), direct ink writing (177), screen printing (150), blade coating (120), inkjet printing (103), spray coating (63), dip coating (26), slot-die coating (8), electrospinning (1) |
| `substrate` | glass, silicon, PET, free-standing, ITO, PI, PDMS, PEN, metal, paper, textile, FTO |
| `specimen_geometry` | film, printed trace, fiber, vertical junction, gel, bulk, porous |
| `measurement_method` | four-point probe, two-point probe, Hall, DC, impedance (direction in `measurement_direction`: in-plane / out-of-plane) |

In optimization loops, keep the text fields fixed at canonical values and vary the numeric fields, the
additive doses and `process_stages`.

### Fixed options and additive structures (strict mode, SMILES)

**Strict mode.** `predict(experiment, strict=True)` (also `predict_many` and `normalize`) runs the model's own
category parser on `polymer_grade`, `deposition_method`, `substrate`, `specimen_geometry`, `measurement_method`
and the additive names, then replaces each by one canonical wording per option. Synonyms of the same option
therefore give bitwise-identical predictions, an unrecognised option raises `ValueError` with the allowed
values, and the result carries `"notes"` listing every rewrite. `pedot_pss.vocabulary()` returns the options
(46 identified additive chemicals, from DMSO and ethylene glycol to acids, surfactants and polymers). Numbers
written inside `deposition_method` are not used in strict mode; give such settings in `process_stages` or
`extra_scalar_features`. Without `strict`, inputs are used as written (v0.1.1 behaviour).

**Additives by structure.** `additive_smiles` is an optional list aligned with `additive_names` (one SMILES or
`null` per additive; needs RDKit: `pip install '.[chem]'`). The model describes additives by identified chemical
and by five RDKit descriptors of the identified molecules (molecular weight, Crippen logP, TPSA, H-bond donors and
acceptors).

- A SMILES of a listed molecule (for example `CS(C)=O`) resolves to that chemical and predicts exactly as its name.
- A molecule that is not in the training chemistry is added for the prediction with its own descriptors, computed
  with the same RDKit calls as the bundled table, and the result's `"notes"` say so.

```python
pedot_pss.predict(dict(experiment, additive_names=["sulfolane"], additive_smiles=["C1CCS(=O)(=O)C1"]))
```

Unlisted molecules are an extrapolation in descriptor space. In the spin-coating example with 5 vol% additive the
model ranks them sensibly (polar aprotic sulfolane and DMAc about 230 S/cm, nonpolar o-xylene and hexane about
80 S/cm; DMSO 533 S/cm; no additive 0.45 S/cm), but the absolute values for nonpolar molecules are too high: the
literature behind the model is dominated by polar secondary dopants, so "an additive is present" itself raises the
prediction. Use unlisted-molecule predictions to compare candidates, and confirm with experiments.

`PEDOT_PSS_PREDICT_THREADS` (default 1) sets the threads of the fitted LightGBM and ExtraTrees learners;
one thread gives bitwise-repeatable predictions.

```python
import json
from pathlib import Path
from pedot_pss import predict, predict_many

experiment = json.loads(Path("examples/polybot_recipe.json").read_text())
prediction = predict(experiment)
print(prediction["conductivity_S_cm"])

# The loaded model is reused within this process.
predictions = predict_many([experiment, experiment])
```

The command-line interface accepts a dictionary or list of dictionaries:

```bash
pedot-pss experiments.json --output predictions.json
```

`--model PATH` selects another complete fitted model with the same input
contract. Prediction does not fit a model or transmit experimental inputs.

## For agent developers

The package is meant to be called as a black-box tool inside optimization or agent loops.

- **Call:** `pedot_pss.predict(experiment)` takes one experiment dictionary and returns
  `{"conductivity_S_cm": float, "log10_conductivity_S_cm": float}`. `pedot_pss.predict_many(list_of_experiments)`
  returns a list in the same order; prefer batches inside a loop.
- **Tool schema:** [`input_schema.json`](input_schema.json) is a JSON Schema for the experiment dictionary and can
  be used directly as the parameter schema of an LLM tool/function definition.
- **Speed (laptop CPU):** loading the model takes about 2–3 s (the very first import after installation takes
  about 30 s while Python compiles the package); a batch of 19 recipes then takes about 1.5 s. No GPU and no
  PyTorch are needed.
- **Deterministic:** the same input always returns the same prediction.
- **Determinism:** pass `strict=True` to map every categorical field to a fixed option (see
  "Fixed options and additive structures").
- **Errors:** unknown field names, outcome or provenance fields (DOI, paper IDs, measured conductivity, SD,
  coverage, ...), malformed numbers or units and unsupported process-stage kinds raise `ValueError`/`TypeError`
  with a message naming the offending field. Use `null` or omit a field when a condition is unknown; do not use 0.
- **Scope:** predictions are most reliable inside the range of formulations and processes covered by the training
  data (literature PEDOT:PSS films plus the Polybot blade-coating platform). Treat predictions far outside it as
  guidance for which experiment to run next, and update with real measurements.

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

There is **one dictionary format with 24 optional fields**. Values include
numbers, material/process descriptions, additive lists and ordered process
stages. The caller supplies experimental conditions; the model constructs its
internal feature vectors.

| Type | Fields |
|---|---|
| Material and amount descriptions | `polymer_grade`, `stock_solids_scope`, `pedot_pss_ratio_raw`, `dose_basis` |
| Process descriptions | `deposition_method`, `substrate`, `specimen_geometry`, `post_treatment`, `process_sequence` |
| Planned measurement descriptions | `measurement_method`, `measurement_direction`, `measurement_quantity` |
| Numeric controls | `stock_solids_wt_percent`, `dry_temperature_C`, `dry_duration_min`, `anneal_temperature_C`, `anneal_duration_min`, `post_treatment_temperature_C`, `post_treatment_duration_min`, `measurement_temperature_C` |
| Lists | `additive_names`, `additive_amounts_raw`, `extra_scalar_features`, `process_stages` |

Supply known conditions. Omitted fields or `null` mean unknown; known zero
remains zero. At least one real planned condition is required. More complete
recipes identify the intended experiment more clearly.

Scalar temperatures use °C and scalar durations use minutes. An extra scalar
is `{ "name": ..., "value": ..., "unit": ... }`, chosen from the 56 allowed
experimental controls. Structured `process_stages` preserve order; their
durations use seconds and temperatures use °C. Explicit stages take priority
when supplied. Unit conversion at the public boundary applies to the following
**extra scalars** only:

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

Their supported alternatives are converted to the units above. Other allowed
extra-scalar units retain the supplied representation; accepting a unit does
not imply universal canonical conversion. Flat fields and structured stages
already require the units named in their field definitions.

Numeric controls require finite JSON numbers. Numeric strings and booleans are
rejected, durations cannot be negative, temperatures must exceed −273.15°C,
and stock solids must be between 0 and 100 wt%. Additive names and amounts use
equal-length lists; use `null` for an unknown amount. Additive amount strings
retain the concentration and basis, for example `"4 vol%"`.

See [input_schema.json](input_schema.json) and the actual source-free recipe in
[examples/polybot_recipe.json](examples/polybot_recipe.json).
`pedot_pss.normalize(experiment)` displays the normalized 24-field dictionary.
The Python API enforces dimensional extra-scalar checks and paired additive
checks in addition to the structural JSON schema.

DOI, paper/source/series/sample IDs, measured conductivity or its SD, measured
coverage and characterization of the finished film are excluded. Unknown
top-level fields are rejected. Do not place identifiers or outcomes inside
experimental description strings.

Input version `experiment-input-v3-text-clean-1012` also removes identifier,
outcome and literature-extraction clauses from experimental text before
feature construction and encoding. Retained clauses describe planned
conditions and their explicit quantities.

## Output

```json
{
  "conductivity_S_cm": 12.3,
  "log10_conductivity_S_cm": 1.0899051114393979
}
```

These numbers illustrate the output format; they are not a model evaluation.

## Included model structure

The port retains the original model's global/default base learners, meta
models and fusion, global recipe kernel and six late residual heads: word
Ridge, numeric/text Ridge, nonlinear text LightGBM, full-feature LightGBM,
character recipe kernel and frozen MatBERT with a fitted rank-64 head. Physics
includes two-phase GEM transport, interface/anisotropy/bypass terms and the
original thermal-competition P correction. Acid, salt, PEO, SWCNT and pure-DAE
branches activate only where explicit experimental inputs support their
defined protocols.

The fitted predictor is refitted on the cleaned population (6,244 points) and wrapped by a final
stacking layer: the seven late residual heads are summed with frozen non-negative weights (non-negative
least squares on training-fold deltas, shrunk by 0.5: word 0.066, nonlinear 0.181, the others 0)
instead of the sequential scalars; prediction-time physics guards return the parent state for salt
and filler inputs outside their training support and apply the PEO law only inside its training
composition range (the pure host, phi = 1, counts as inside); P uses the physical correction refitted
on the combined M. Polybot rows pass through unchanged.

The Polybot platform branch (PH1000, blade coating on silicon, film, with all nine
planned controls) is a two-state physical model: a percolating-film state given by
the unchanged process-GEM closure plus a bounded SVR residual, and a dewetted /
non-percolating state, mixed by a DMSO-gated failure probability. The probability
combines two Gaussian-process classifiers on film-formation coordinates computed
from the controls: the blade-coating regime number (capillary number and Antoine
water evaporation flux at the coating temperature) and the solvent-wash contact
time and temperature. It was fitted on all 75 Polybot runs.

Source-local routes use the original global fallbacks; unavailable observed
features and source bias are removed. Learned parameters and late scalar
weights are refitted inside training data. This retains the usable original
mathematical modules without claiming every original M branch or its former
weights remains unchanged. See [MODEL_CARD.md](MODEL_CARD.md).

The package contains model code, a fixed molecular-descriptor registry,
input/output examples and the pinned fitted model artifact. It contains no data
corpus, private data loader, cached training embeddings or private workflow
files. Prediction requires no private project directory.

Historical retrospective R² values near 0.95/0.97 that used same-assay
conductivity SD or measured coverage are not scores for this experiment-only
predictor. A numerical output outside evaluated recipe ranges does not itself
establish extrapolation accuracy.

## MatBERT attribution

MatBERT is the frozen materials-language encoder released by the Lawrence
Berkeley National Laboratory Natural Language Processing project (LBNLP).
The setup command fetches assets from the author's Ceder-group S3 distribution.
The original MIT notice is preserved in
[third_party/MatBERT_LICENSE.txt](third_party/MatBERT_LICENSE.txt) and accompanies
the downloaded assets. Only the downstream rank-64 projection and prediction
head are included in this surrogate's fitted state; the encoder is not
retrained.
