# Model card: PEDOT:PSS physics + ML hybrid conductivity surrogate (v0.2.0)

**Task.** Predict the electrical conductivity (S/cm, reported as log10) of a planned PEDOT:PSS experiment from its
material, recipe, processing and planned measurement conditions. Intended use: a surrogate for screening and
ranking candidate experiments (Bayesian optimization, active learning, agent loops), followed by real measurements.

**Inputs.** One experiment dictionary (24 optional fields) plus optional SMILES for molecules outside the bundled
chemistry table. Missing input stays missing. Provenance identifiers, measured conductivity or its SD, measured
coverage, film thickness and post-fabrication characterization are rejected.

**Representation.** Every chemical and material enters through descriptors, never through its name:
- additives and post-treatment chemicals: 35 RDKit descriptors and functional-group counts, structure classes and a
  dose saturation, aggregated per recipe (post-treatment rinses kept apart from the active treatment);
- grade and substrate: nominal material properties from supplier datasheets and handbooks;
- deposition method, specimen geometry and measurement method: fixed categories; numbers in fixed units.
Synonyms are rewritten to one canonical wording before training and prediction. The model contains no text
embedding or bag-of-words component.

**Results** (pooled held-out predictions on fixed random-row five folds over the dataset of 6,244 points: 6,169 from
429 papers and 75 Polybot runs):

| Output | All 6,244 R² | Literature 6,169 R² | All RMSE (dex) |
|---|---:|---:|---:|
| P (default) | 0.960 | 0.961 | 0.365 |
| M | 0.960 | 0.961 | 0.364 |

Final-model fit on the 75 Polybot runs: R² 0.962 (training fit, not held out). Unseen chemistry (all rows with a
given additive held out together): R² 0.48 on those rows with this representation against 0.42 for the same model
with name identities. The representation was chosen among 12 full-model variants on these folds; the paired
bootstrap 95% interval of its change against the identity-based control was within [-0.002, +0.003].

**Architecture.**
- Global base: ten GEM + residual-learner candidates (LightGBM, CatBoost, ExtraTrees), a calibrated GEM physics
  reference, 17 meta models and a Ridge fusion. Residual learners are also trained on copies of the rows with the
  chemical identity coordinates masked (identity dropout), so the descriptor path is learned everywhere.
- Physics: McLachlan two-phase general effective medium with a process-dependent low-conducting endpoint, a positive
  interface resistance, anisotropy and bounded bypass (DomainTransport). Acid (process gate), salt (applied
  concentration, contact time and temperature), PEO (percolation in the nominal host volume fraction), SWCNT
  (parallel path) and pure DAE (insulating prior) branches act only where the inputs define their protocol and only
  inside the range covered by training.
- Late layer: a global recipe kernel (256 Gaussian anchors, Ridge) on the encoded vector is fitted as a possible
  late correction; its non-negative weight on the full training data is 0 (0 to 0.32 across the five folds), so the
  released model has no late correction.
- P: two cumulative Arrhenius clocks (thermal ordering E = 0.35 eV, damage E = 0.40 eV) modify the interface
  resistance and the phase conductivities of the physical baseline, which is then re-solved; recipes outside the
  conventional electronic PEDOT:PSS domain keep P = M.
- Polybot platform branch (PH1000, blade coating on silicon, film, nine planned controls): a percolating-film state
  (process GEM plus a bounded SVR residual) and a dewetted state, mixed by a DMSO-gated failure probability from two
  Gaussian-process classifiers on a blade-coating regime number and the solvent-wash contact time and temperature.

**Training.** All learned parameters (scalers, physical coefficients, learners, meta/fusion, the late-layer weight,
physics support ranges, P parameters) are fitted inside the training data of each outer fold; inner three-fold
splits create the stacking covariates. The chemistry table (PubChem structures, RDKit descriptors) and the material
table are fixed resources that read no labels. The deployed model is fitted once on all 6,244 points.

**Limitations.** Random-row folds measure interpolation within this corpus, not new-laboratory or new-chemistry
extrapolation; molecules outside the training chemistry are an extrapolation in descriptor space. About 12% of
additive mentions have no single structure (vague names, commercial mixtures) and enter through class flags only;
polymer descriptors describe the repeat unit. The physics layers still recognize a few chemicals by identity
(DMSO/EG process coordinates, acid and polar-solvent treatments, the acid/salt/PEO/SWCNT/DAE branches); a new
molecule enters the data-driven layers through its descriptors only. Material descriptors are nominal values, not
batch- or surface-treatment-specific. Effective physical parameters do not uniquely identify microscopic material
properties.

**Distribution.** The complete fitted state (`model.joblib`) is the asset of GitHub Release v0.2.0;
`python -m pedot_pss setup-model` installs it. No corpus or per-point evaluation data is distributed. Standalone
inference was verified in a fresh environment with network access and reads of the development tree blocked.
