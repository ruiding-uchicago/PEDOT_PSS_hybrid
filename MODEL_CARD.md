# Experiment-only original-module M/P port

**Complete fitted model with independently checked five-fold results.**

Standalone inference has been verified in a fresh Python 3.12 environment on macOS:
the installed package reproduces the private development model on 19 recipes
(all seven physics branches, Polybot and literature rows, new recipes) to within
1e-6 log10, with network access and reads of the development tree blocked. The
model asset has not been published yet.

After removing 536 low-quality literature points (8%; rule `keep_v2_cap08`, which
used out-of-fold residuals plus independent extraction evidence), pooled five-fold
held-out results on 6,244 points:

| Output | All 6,244 R² | Literature 6,169 R² | All RMSE (dex) |
|---|---:|---:|---:|
| P (default) | 0.960 | 0.961 | 0.364 |
| M | 0.960 | 0.961 | 0.363 |

Final-model fit on the 75 Polybot runs: R² 0.962 (training fit, not held out).
Selection-aware estimates: P 0.960 / M 0.960 with combiner, guards and P cap chosen
leave-one-fold-out.

Intended use: a low-fidelity surrogate for initial PEDOT:PSS experimental
screening. Planned material, recipe, processing and measurement conditions go
in; predicted conductivity in S/cm and log10(S/cm) comes out. Missing input
remains missing. Fixed molecular descriptors describe input molecules, rather
than measurements of the fabricated film.

The port preserves original global/default learner families and meta/fusion
equations, the source-free global recipe kernel, six late residual heads and
supported physical corrections. The heads are word Ridge, numeric/text Ridge,
nonlinear text LightGBM, full-feature LightGBM, character kernel/Ridge, and
frozen MatBERT/rank-64/LightGBM. Each late scalar is learned within outer TRAIN
using three inner folds and constrained to [-1, 1]. Each signed layer weight is refitted inside TRAIN; historical posthoc weights
are not used for prediction.

The physical P correction retains original DomainTransport two-phase GEM,
interface, anisotropy and bypass calculations; removes duplicated thermal
columns; and reuses the original thermal-competition common-14 correction.
Known process stages supply thermal history. Parameters, scalers and linear
corrections use TRAIN only. Unsupported or insufficiently specified protocols
retain their parent prediction. Acid, salt and PEO, plus nominal-mass SWCNT
parallel transport and an explicit UV-on pure-DAE prior, use original cores
within input-defined applicability. The DAE prior is -12 log10(S/cm), not a
measured endmember. SWCNT nominal mass is not asserted to be volume fraction.

Polybot platform branch: within PH1000/blade-coating/silicon/film inputs with all
nine planned controls, the prediction is a two-state mixture. The percolating film
state is the unchanged process-GEM closure plus a bounded SVR residual; the
dewetted state is the mean training level of non-percolating films. A DMSO-gated
failure probability combines two Gaussian-process classifiers on film-formation
coordinates derived from the controls (blade-coating regime number from capillary
number and Antoine water evaporation flux; solvent-wash contact time and
temperature). It is fitted on all 75 Polybot runs. Coverage and same-assay SD are
not inputs; they were not used to fit this branch.

Provenance identifiers, same-assay conductivity statistics, measured coverage
and post-fabrication characterization are removed from model input. Original
source-local routes use documented global fallbacks and source bias is
removed. Data-dependent branches without prospective input support cannot
remain unchanged. This retains original mathematical modules without claiming
the entire historical M graph and all old weights are unchanged.

The current input projection is `experiment-input-v3-text-clean-1012`.
Outcome statements, provenance identifiers and literature-extraction wording
are removed before experimental text reaches features or the frozen encoder.

Evaluation population: 6,244 conductivity points after data cleaning (6,169 literature points from 429 papers and 75 Polybot runs).
The five outer row folds are fixed. Targets and reported errors use log10
conductivity in S/cm. The table reports complete held-out five-fold metrics, not training accuracy. Random-row evaluation does not measure new-laboratory or control-range
extrapolation. Any grouped or control-range results must be reported separately
with their split definitions.

Previous retrospective all-data R² around 0.95 and Polybot R² around 0.97 belong
to a system using additional unavailable information. They are not scores for
this experiment-only predictor.

The standalone release requires the complete fitted `model.joblib` state only.
Since v0.1.1 prediction does not run the frozen MatBERT encoder: the joint late
combiner gives the MatBERT head weight 0 and every stage after the last weighted
one is skipped, which leaves the predictions bitwise identical to v0.1.0. It builds features and intermediate
predictions internally, requiring no literature corpus, training-row lookup
or private source directory. The encoder receives projected experimental
text and is frozen. Effective physical parameters do not uniquely identify
microscopic material properties.

Since v0.1.2, `strict=True` maps the categorical inputs to fixed options and `additive_smiles` lets an additive be given
by structure (unlisted molecules enter through RDKit descriptors only, as an extrapolation).

The default published output is P. The complete fitted state is distributed
in GitHub Release v0.1.0; `setup-model` installs it once. No corpus or
per-point evaluation data is distributed. The frozen MatBERT backbone
has 12 transformer layers, 12 attention heads per layer and hidden size768;
its rank64 projection uses TRAIN inputs and its regression head uses TRAIN labels.
