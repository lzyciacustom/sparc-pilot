# SPARC pilot — core findings (H-optimus-1; 28 Oct 2026)

This is the current summary. The H-optimus-0 / 18-group tables in [RESULTS.md](RESULTS.md) are the earlier stage-2 record. Tables behind sections 3 and 7–10 are in `results/tables/{h1,headroom,composition,mdacc}/`; figures in `results/figures/stage3_composition/`.

All numbers are held-out, patient-group-level means over **16 CRC patient groups (47 sections)**; 2 groups whose patients also contributed training-pool slides are excluded (all-18 results are in docs/AUDIT.md and lead to the same conclusions). Training on HEST-1k (~376 slides per fold, 6 folds) unless stated otherwise.

## Metric definitions
- **Raw r**: Pearson correlation between predicted and measured program activity across spots of one section, averaged over 36 programs, then over seeds, then over sections within a patient group.
- **Depth-adjusted r**: the same after regressing log(1 + total UMI) out of *both* prediction and measurement within each section.
- **Random-gene-set null**: an identical model trained on size-matched random gene sets (ssGSEA: 3 draws; ULM: 1 draw).
- **Seed noise**: SD of group-level differences between seeds = 0.027 (ULM); grey band in figures = ±0.032 (2 SD/√3).
- Tests: paired Wilcoxon over the 16 groups; "k/16" = groups with a positive difference.
- Each model is scored against its own target type (ULM model vs measured ULM scores; ssGSEA model vs measured ssGSEA scores).

## 1. Raw correlation is mostly non-specific (SPARC-style ssGSEA targets)
| | H-optimus-0 | H-optimus-1 |
|---|---|---|
| Raw r, curated vs random | 0.447 vs 0.399 (random = 89%) | 0.466 vs 0.408 (88%) |
| Depth-adjusted r, curated vs random | 0.243 vs 0.058 (16/16) | 0.244 vs 0.067 (16/16) |

## 2. The largest lever is the training target (depth-adjusted r)
| Change vs ssGSEA targets | H-optimus-0 | H-optimus-1 |
|---|---|---|
| ULM targets | +0.102 (15/16, p = 6e-5) | **+0.109 (16/16, p = 3e-5)** |
| Depth-regressed ssGSEA | +0.095 (15/16) | +0.098 (15/16) |
| MLM targets | +0.029 (12/16, n.s.) | +0.034 (12/16, p = 0.02) |
ULM vs depth-regressed ssGSEA: +0.008 (10/16, n.s.) — the gain comes from removing the shared, depth-linked component, not from ULM itself.

## 3. The image encoder is not the bottleneck (H-optimus-1 minus H-optimus-0, depth-adjusted r)
| Target | HEST-1k training | CRC-only training (~46 slides/fold) |
|---|---|---|
| ssGSEA | +0.001 (9/16, p = 0.74) | +0.028 (11/16, p = 0.006) |
| ULM | +0.008 (8/16, p = 0.38) | +0.027 (11/16, p = 0.013) |
| Random sets | −0.000 to +0.013 | −0.013 to +0.004 |
With full training data the encoder change is within seed noise; raw r rises 0.01–0.05 but random-set raw r rises similarly.

## 4. Other components (H-optimus-0, ULM targets unless stated)
| Change | Δ depth-adjusted r |
|---|---|
| 8× more training data (HEST-1k vs CRC-only) | +0.020 (12/16, p = 0.07) |
| No neighbours (centre tile only) | −0.019 (2/16, p = 1.5e-4) |
| CONCERT-style attention bias, GP kernels (k = 16/48), 448 µm field | −0.009 to +0.002 (n.s.) |
| Gaussian head, gene decoder, auxiliary gene loss | −0.003 to +0.011 (n.s.) |
| Slide z-scored targets | −0.027 (2/16, p = 8e-4) |

## 5. Signal is not explained by depth, nuclei or composition (ULM, H-optimus-0)
| Adjustment | curated vs random |
|---|---|
| Depth | 0.345 vs 0.091 (16/16) |
| + nuclear morphology (HoVer-Net; FFPE; 9 groups) | 0.321 vs 0.073 (9/9) — 91% retained |
| + cell-type composition (NNLS, 12 types; CRC-trained model) | 0.143 vs 0.029 (16/16) |

## 6. Niches and mouse (H-optimus-0)
- Niches (PCA + k-means, K = 4): predicted vs measured ARI 0.177, above clustering H&E features directly (0.148; 13/16, p = 0.02) and a random-set model (0.135; 13/16, p = 4e-4).
- Mouse (22 public sections, ULM on 1:1 orthologs): human-trained 0.10; mouse-retrained 0.26 within study; 0.10 across studies (no animal IDs in HEST).

## 7. Headroom: how reliable are the measured targets? (H-optimus-1, no training)
Split-half reliability of the measured program scores on the CRC test sections, Spearman–Brown corrected; ceiling = maximum depth-adjusted r any predictor could reach against the measured target.
| Target | Ceiling, UMI split (technical noise) | Ceiling, gene split | Achieved (HEST-1k, 3 seeds) | Achieved / ceiling |
|---|---|---|---|---|
| ULM | 0.747 | 0.645 | 0.353 | 47% (55%) |
| ssGSEA | 0.756 | — | 0.244 | 32% |
| Random sets (ULM) | 0.369 | 0.117 | 0.095 | 26% |
- ULM and ssGSEA targets are equally reliable, so ULM's advantage is not lower target noise: its depth-adjusted variation is more image-related.
- Across programs, achieved accuracy tracks the ceiling (r = 0.80); stromal/ECM programs are near 0.7 of a ~0.95 ceiling, immune/metabolic programs reach 0.2–0.3 of 0.6–0.85.
- The ceiling counts technical noise only; expression that is invisible in H&E is not counted, so the true headroom is smaller than the gap shown.

## 8. Composition-aware SPARC-Map (CRC-only training, H-optimus-1, 3 seeds)
- **b_M (reference baseline):** 80,000 simulated pseudospots of known composition from GSE132465 (12 types; 100–100,000 UMI), ULM-scored; MLP from (composition, log depth) to program scores. Held-out median R²: 0.75 (composition + depth), 0.34 (composition only), 0.15 for random sets. Calibrated per program on training spots only.
- **Residual (reference-relative):** δ = y − b(π_deconv, depth), composition and depth removed; defined relative to a fixed reference, not a claim about pure within-cell state.
| Depth-adjusted r vs measured ULM | Value | vs direct |
|---|---|---|
| Direct ULM model | 0.348 | — |
| Composition only, b(π̂) | 0.312 | −0.036 (5/16, p = 0.05) |
| Two-stage (separate π̂ and δ̂ models) | 0.376 | +0.028 (13/16, p = 0.004) |
| Joint (one model, b(π̂) + δ̂) | 0.378 | +0.031 (14/16, p = 2e-4) |
- Joint vs two-stage: +0.002 (8/16, p = 0.60) — the gain comes from the decomposition, not from joint training; the simpler two-stage model is the default unless joint training helps on new cohorts.
- Residual accuracy (r with measured δ): two-stage 0.273, joint 0.264, residual implicit in the direct model 0.199 (both 16/16); random gene sets 0.092 (real − random +0.18, 16/16).
- Gene overlap control: 27% of the original 600 deconvolution markers were program genes. With a gene-disjoint marker panel (0 overlap), residual accuracy is 0.287 (vs random 0.092, 16/16; 13/16 groups higher than with the original panel) — shared genes do not inflate the residual result.
- Composition prediction from H&E: mean r(π̂, π) = 0.385 across 12 types.

## 9. Niche decomposition and a stage contrast (measured data; exploratory)
- **Niches from H&E only** (k-means on neighbourhood-averaged H-optimus-1 features; 4 niches present in ≥ 8 groups tested). Enrichment splits exactly into composition + depth + residual. Of 100 significant niche × program enrichments (FDR < 0.1), composition is the larger part in 40 and the residual in 20; 39/41 residual hits replicate with the gene-disjoint panel.
- **Caveat (important):** random gene sets also show reproducible residual niche structure (71/144 cells at FDR < 0.1 vs 41 for curated programs; against a leave-one-out random null, 45 vs 75). H&E-predicted residual niche patterns match measured ones equally for curated and random sets (r = 0.611 vs 0.608). Cross-patient replication of a residual niche pattern is therefore not, by itself, evidence of program-specific biology; condition contrasts with random-set nulls are needed.
- **Stage contrast (COLON MAP, pathologist border vs core):** 16 carcinoma patients with both regions (6 stage III/IV vs 10 stage I/II; 28 sections; permutation within MSI strata). No component passes FDR; residual 3/36 programs at p < 0.05 vs 1/36 for random sets (chance ≈ 1.8). H&E-predicted border-vs-core residual agrees with measurement at per-patient r = 0.70 (random sets 0.54; 10/16, p = 0.05).
- Power: patient-level standardised effect d = 1.0 needs ~16 animals or patients per arm (80% power, α = 0.05); d = 0.8 needs ~25.
- Multiplex immunofluorescence check not possible: the OSF quantification files for COLON MAP are empty; per-cell MxIF exists only as raw images on the HTAN portal.

## 10. External cohort: CRC liver micro- vs macrometastases (MD Anderson, GSE294385; measured data)
- Data: Liu et al., Cancer Cell 2026. 32 FFPE Visium sections (24 liver metastasis, 8 primary colon; 5 lung sections excluded — no lung reference), 294,852 spots with the authors' pathology layers. No full-resolution H&E is deposited, so only the measured decomposition is possible. GEO does not give patient IDs; the sample → patient map was transcribed from the paper's Table S1 (`metadata/mdacc_patient_map.csv`). Patient-level results are primary; the section-level results below them are a sensitivity analysis.
- Reference extended to 17 types: GSE132465 (12 colon types) + GSE115469 liver atlas (hepatocyte, cholangiocyte, LSEC, Kupffer, stellate; counts recovered exactly from the log2-normalised matrix). b_M held-out R²: 0.74 curated, 0.15 random. Calibration fitted on non-neoplastic liver/colon spots only.
- **Patient level (primary analysis)** (Table S1 mapping; 29 sections from 11 patients; 3 IX_566xx sections not in Table S1 excluded; sections averaged within patient):
  - Micro- vs macrometastasis stroma, 10 patients: PD-1-high CD8 program total +1.24 SD; residual +1.06 (10/10 patients; z vs random 3.96, largest random |z| 1.32; q = 0.035), composition +0.24. Gene-disjoint panel: residual +1.03, 10/10, z 3.83, q = 0.059. Collagen formation residual −0.46 (lower in 10/10; q = 0.035), EMT residual −0.30 (lower in 9/10; q = 0.035), IFN-γ +0.48 (8/10) and TNFα/NF-κB +0.55 (7/10; both q = 0.044). 11 curated residual effects exceed the largest random-set effect; the leave-one-out random null still gives 6 residual hits at q < 0.1 (vs 14 curated).
  - Micro- vs macrometastasis tumour, 9 patients: composition-dominated; no curated residual effect at q < 0.1 (random leave-one-out: 6).
  - Primary vs liver tumour, 5 vs 11 patients (unpaired): residual 14 programs at q < 0.1 vs 7 for random sets (e.g. neutrophil degranulation −0.82, MHC-I presentation −0.70 in primaries); cross-organ, not competence.
- This matches the published description (immunosuppressive, T-cell-exhausted niche around micrometastases; myofibroblast encapsulation of macrometastases) and adds that the differences are mainly cell state, not cell abundance.
- **Section level — micro- vs macrometastasis stroma (12 sections, paired):** the PD-1-high CD8 T-cell program (GSE26495) is higher around micrometastases by +1.08 SD; composition contributes +0.22 and the residual +0.92 (12/12 sections; z vs random sets 3.6, largest random |z| 1.21; q = 0.018; gene-disjoint deconvolution z 3.5, 12/12). T/NK fraction changes by only +0.04. Residual IFN-γ (+0.46) and TNFα/NF-κB (+0.54) are higher; collagen formation (total −0.64; residual −0.45) and EMT (−0.45; residual −0.29) are lower. 12 curated residual effects exceed the largest random-set effect.
- **Section level — micro- vs macrometastasis tumour (9 sections):** composition-dominated (tumour fraction −0.25, hepatocyte +0.08: dilution of small foci by liver); no residual effect at q < 0.1. Cell-cycle residual +0.34/+0.38 (n.s.), so the published quiescence finding is not testable at 55 µm spot resolution here.
- **Section level — primary vs liver tumour (8 vs 22 sections, unpaired):** mostly residual (15 programs at q < 0.1 vs 10 for random sets) — a cross-organ contrast that cannot be read as metastatic competence.
- Caveats (section level): section-level replication; a leave-one-out random null still yields 6 residual hits at q < 0.1 in the stroma contrast (vs 13 curated), so q-values are optimistic — the comparison with the full random-set range is the conservative test.

## Implications for the proposal
- The decomposition is technically sound (exact, calibrated, robust to shared genes) and improves accuracy (+0.03) and residual prediction (+0.07).
- Joint training is not yet justified over a two-stage pipeline; state this as a test, not a claim.
- Within-tumour niche patterns alone cannot isolate program-specific residual biology; Aim 2 should be framed around condition contrasts (metastatic vs non-metastatic, perturbed vs control) with random-gene-set nulls and adequate sample sizes — the design the Yilmaz/Bai models provide.
- An external metastasis cohort (section 10) shows the decomposition separating abundance from state in a published niche contrast; PETACC8 (48 stage III primaries, 14 relapsed; embargoed to Oct 2027) is the planned human outcome cohort.

## Not yet repeated with H-optimus-1
Spatial-model variants, engineering upgrades, nuclear adjustment, the K = 4 niche ARI analysis, mouse. (Sections 7–9 use H-optimus-1.)

## Known deviations from SPARC (see docs/METHODS.md, docs/AUDIT.md)
Training code and slide list not released (architecture imported from aziz-ayed/SPARC a046572; recipe from the paper); 36 of SPARC's programs (7 custom signatures missing); bs 1024 / lr 2e-4 / bf16 for all conditions; relative coordinates scaled by median 1-NN distance; per-section rather than pooled-patch Pearson; CRC test set rather than HEST-Bench; random sets size-matched but not expression-matched; official SPARC-Map weights not yet compared.

## Where things are in this repository
| Content | Path |
|---|---|
| H-optimus-1 per-slide results (all runs) | `results/raw/hest1k_h1_long.parquet` (`scripts/07_analysis/collect_h1_runs.py`) |
| H-optimus-0 vs H-optimus-1 (section 3) | `results/tables/h1/h0_vs_h1_{hest1k,crconly}_16groups.csv` (`scripts/07_analysis/compare_h0_h1.py`) |
| Noise ceiling (section 7) | `results/tables/headroom/` (`scripts/08_headroom/`) |
| b_M fit, composition-aware model, niches, COLON MAP stage (sections 8–9) | `results/tables/composition/` (`scripts/09_composition/`) |
| MDACC decomposition (section 10) | `results/tables/mdacc/` (`scripts/09_composition/mdacc_*.py`) |
| Figures | `results/figures/stage3_composition/` (`scripts/07_analysis/figures_stage3.py`) |
H-optimus-1 features (388 slides, 828,636 patches) and per-spot predictions are not in git (size); `scripts/04_hest1k/h1/` regenerates them.
