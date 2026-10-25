# Pipeline audit and known limitations

The audit was run after all experiments, before packaging. Checks were run against the as-run split files and the raw result tables.

## Checks that passed

| Check | Result |
|---|---|
| CRC test folds disjoint at patient-group level | ✓ Each of 18 groups is in exactly one fold |
| Label normalisation statistics (median/IQR) computed on training slides only | ✓ (`sparcmap_lib.robust_stats` called on `trs`) |
| Validation group drawn from in-fold training CRC groups only | ✓ |
| CRC test candidates present in the HEST-1k pool removed before training | ✓ All 54 overlapping IDs are excluded at runtime |
| NaN metrics in final result tables | 0. A first random-target run had NaN on 81% of folds: random-set ULM produced fully NaN columns on a few slides (e.g. MISC69), and `sqrt(0)` in the Pearson loss gave infinite gradients. Fixed in `sparcmap_lib.pearson_loss` (eps inside sqrt, masked columns skipped); all random-target conditions were rerun (debug scripts in `attic/`). Separately, `fix_ulm.py` recomputed stage-1 HEST-Bench ULM/MLM targets that had failed. |
| Wave-to-wave reproducibility of the base model | ✓ ULM depth-res 0.3465 (wave 2) vs 0.3461 (wave 1) |
| Analysis scripts regenerate the reported numbers | ✓ `hest1k_stats.py`, `mouse_stats.py`, `nuclei_stats.py` and `figures.py` were re-run from `results/raw/` |

## Finding 1: training-pool leakage (minor; does not change conclusions)

**Problem.** Five training-pool slides are from the same patients or tissue blocks as CRC test groups:

- MISC52–55: COLON MAP sections `8270_AS_1–5`, the same block as test group `COLONMAP_8270_AS`.
- TENX153: healthy colon from Patient 1 of the 10x immune dataset, the same patient as test group `TENX_immune_P1`.

**Root cause.**
- The pool was filtered by `disease_state` (Tumor/Healthy) and only excluded exact slide IDs.
- These sections are tagged differently from their test-set siblings, so the filter did not catch them.

**Sensitivity analysis.** All main comparisons were re-tested excluding the two affected groups (16 groups); see `results/tables/audit_leakage_sensitivity.csv`.
- Every effect keeps its sign and approximate size.
- ULM vs ssGSEA: +0.102 (15/16).
- ULM real vs random: +0.254 (16/16).
- HEST-1k vs CRC-only (ULM): +0.020 (12/16, p = 0.07, previously p = 0.04).
- The leaked groups did not gain more from HEST-1k training than other groups.

**Fix for future runs.** `python scripts/00_data/build_splits.py --strict` removes these slides. It also excludes the pool by patient/block, not only by slide ID.

## Finding 2: null gene sets are size-matched, not expression-matched

**Problem.** Random sets were drawn uniformly from the 13,586-gene universe. Real programs are enriched for highly expressed, spatially variable genes, so part of the real-minus-random gap may come from expression level rather than biology.

**Mitigations.**
- Depth residualisation removes the largest expression-level confound.
- ULM's per-spot regression is less sensitive to mean expression than ssGSEA ranks.

**Fix for future runs.** An expression-binned (deciles of mean log expression) null is planned.

## Finding 3: composition analysis is an upper bound on "composition-driven" signal

**Problem.**
- NNLS proportions are estimated from the same spots' expression. Marker genes shared between a program and the reference signatures inflate composition R², e.g. ECM genes in the fibroblast profile.
- Residual targets are computed per slide, which removes slide means.
- RCTD was only run on 3 slides as a sanity check. Agreement with NNLS is reported in `stage2_inputs/rctd_vs_nnls_*.csv`.

**Interpretation.** Composition R² is best read as an upper bound, and residual predictability as a lower bound, on cell-state signal.

## Finding 4: mouse split cannot be animal-level

**Problem.** HEST does not provide animal IDs. Under the slide-level scheme, serial sections of one animal may be split across train and test.

**Evidence.** The leave-dataset-out scheme shows no gain from mouse retraining. The +0.16 slide-scheme gain should therefore be treated as within-dataset (optimistic) until it is retested with animal-level splits on lab-generated data.

## Finding 5: nuclear segmentation scope

**Scope limits.**
- FFPE slides only (45 of 58). Frozen sections were too blurred, and a simple threshold-based nucleus counter failed.
- A random 10% of spots per slide (16,143 spots).
- 11 of 18 groups.

**Measurement limits.**
- Segmentation covers only the central 41 µm of each 55 µm Visium spot.
- PanNuke type calls are not CRC-specific: inflammatory vs immune agreement is ρ = 0.12.

**Interpretation.** The 91% retention figure applies to this subset.

## Other deviations and caveats

**Programs**
- 36 of 43 SPARC programs are available from MSigDB. The 7 custom signatures are not used (listed in `metadata/sparc_programs_msigdb.json`).

**Model**
- SPARC-Map is a reproduction, not the released checkpoint. Relative coordinates are scaled by each slide's median nearest-neighbour distance.
- H-optimus-1 access was pending, so all features use H-optimus-0.

**Data scope**
- Stage 1 (HEST-Bench) is pan-cancer: only 4 slides are READ. Its conclusions on targets and heads were re-tested in stage 2 on 58 CRC slides; its spatial and active-learning conclusions were not.

**Statistics**
- Some secondary conditions have 1–2 seeds (`n_seeds` column).
- There is no multiple-testing correction. Effects smaller than the seed-noise band (±2 SD: ssGSEA 0.07, ULM 0.05 at group level) are reported as null.

**Training budget**
- Wide-output variants (gene decoder, gene auxiliary loss, composition targets) train for a fixed 30 epochs without early stopping.

**Provenance**
- Three sub-analyses (composition decoupling, mouse targets and the HEST-Bench tracks) were run in their own working directories. Their code is included as run. Path differences are noted inline, e.g. `compute_targets_mouse.py`.
