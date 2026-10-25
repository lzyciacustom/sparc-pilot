# Track B — Spatial context and prediction heads: methods and results

## 0. Summary
- **Spatial context (B1).** A CONCERT-style attention bias (distance + H&E similarity + compartment) adds little over the distance-only SPARC-Map baseline, and less than training-seed noise. On ssGSEA targets: raw Pearson +0.0097 (39/57 slides improved, Wilcoxon p = 0.016; 10/16 patient groups, p = 0.13), depth-residualised Pearson +0.0050 (p = 0.11; 13/16 groups, p = 0.029). No gain on MLM targets (+0.0062 / +0.0025, both p > 0.2). The gain is **not concentrated in READ**, and k = 48 **does not help**. Re-training the same baseline with another seed changes ssGSEA depth-residualised Pearson by −0.0196 (p = 0.0013), more than the CONCERT effect.
- **Prediction heads (B2).** A low-rank Gaussian head matches the deterministic head in accuracy and greatly improves CRPS and energy score (CRPS: 15/16 groups improved for ssGSEA, 16/16 for MLM; energy score: 16/16 for both targets). The flow-matching head loses accuracy (raw −0.038 / −0.043). Flow point predictions are less over-smoothed, but flow samples are worse than Gaussian samples on covariance and on the S matrix. Both heads under-cover in **every patient group**: the Gaussian 90% interval covers 0.695 of values for ssGSEA (0.36–0.90 across groups) and 0.771 for MLM.

## 1. Data and evaluation
- 57 Visium whole-transcriptome slides in 16 patient groups, 4 folds from `L.wt_folds()`; no patient group spans train and test. In each fold one patient group from the training folds is held out as an inner validation group for early stopping (same rule as `run_baseline.py`); the criterion is the validation group's mean raw Pearson.
- Targets: precomputed `ssgsea__*` and `mlm__*` scores for the 36 SPARC programs, robust-normalised with training-set median/IQR (`L.robust_stats`).
- Features: H-optimus-0 (1536-d); k = 16 nearest neighbours (k = 48 for the k48 variants).
- Metrics: raw Pearson and depth-residualised Pearson (residuals after regression on log1p(total counts), `L.evaluate`) per slide and program, then averaged over programs. Variants with two seeds are averaged over seeds first. READ (4 slides, 2 patient groups) is reported separately.
- Paired tests: (i) paired differences over the 57 slides, Wilcoxon signed-rank; (ii) patient-group level: slide differences averaged within each group, Wilcoxon over the 16 groups, with the number of improved groups reported. Pairs use only (fold, seed) combinations present for both variants. No multiple-comparison correction.

## 2. B1 — Spatial-context ablation
### 2.1 Models (all 30 epochs, AdamW wd = 1e-2, gradient clipping 1.0, loss = MSE + 0.5·(1 − Pearson))
| Name | Description | Parameters |
|---|---|---|
| mlp | centre-patch-only MLP, 1536→1024→512→256→36 (LayerNorm + GELU + dropout 0.1) | 2.24 M |
| base_k16 | SPARC-Map baseline `L.build_model` (relative-coordinate MLP, 3 transformer blocks, H = 256, 8 heads) | 2.82 M |
| concert_k16 | Same transformer (pre-LN blocks with `scaled_dot_product_attention` and a float bias), keeping the relative-coordinate encoding, plus an additive attention bias per layer and head built from: (1) 16 RBF features of the physical distance between tokens → linear map; (2) 9 RBF features of the cosine similarity of H&E features between tokens (after subtracting the training feature mean) → linear map; (3) compartments: k-means (k = 6) on PCA-32 features fitted on all non-test slides of the fold (training + inner validation), assigned inside the model by nearest centroid, with a K×K×(layers×heads) compartment-pair bias table and a compartment embedding added to each token. The bias output layers are zero-initialised, so training starts from the distance-only model | 2.82 M |
| noDist / noCos / noComp | concert_k16 without (1), (2) or (3) | 2.82 M |
| base_k48 / concert_k48 | k = 48 neighbours | as above |

### 2.2 Results (mean over 57 slides; tables `B1_summary_by_variant.csv`, `B1_paired_tests.csv`)
| Target | Variant | Seeds | raw | depth-res | READ raw | READ depth-res |
|---|---|---|---|---|---|---|
| ssgsea | base_k16 | 2 | 0.3622 | 0.1572 | 0.4353 | 0.2336 |
| ssgsea | mlp | 2 | 0.3732 | 0.1512 | 0.3867 | 0.1932 |
| ssgsea | concert_k16 | 2 | 0.3720 | 0.1622 | 0.4574 | 0.2395 |
| ssgsea | noDist_k16 | 2 | 0.3633 | 0.1569 | 0.4342 | 0.2132 |
| ssgsea | noCos_k16 | 2 | 0.3617 | 0.1496 | 0.4119 | 0.1977 |
| ssgsea | noComp_k16 | 2 | 0.3545 | 0.1474 | 0.4145 | 0.2018 |
| ssgsea | base_k48 | 1 | 0.3638 | 0.1442 | 0.4443 | 0.2036 |
| ssgsea | concert_k48 | 1 | 0.3639 | 0.1476 | 0.4045 | 0.1825 |
| mlm | base_k16 | 2 | 0.2646 | 0.1732 | 0.3654 | 0.2940 |
| mlm | mlp | 2 | 0.2527 | 0.1635 | 0.3424 | 0.2728 |
| mlm | concert_k16 | 2 | 0.2707 | 0.1758 | 0.3353 | 0.2795 |
| mlm | noDist_k16 | 2 | 0.2727 | 0.1766 | 0.3280 | 0.2685 |
| mlm | noCos_k16 | 2 | 0.2674 | 0.1759 | 0.3280 | 0.2720 |
| mlm | noComp_k16 | 2 | 0.2750 | 0.1758 | 0.3498 | 0.2777 |
| mlm | base_k48 | 1 | 0.2715 | 0.1725 | 0.3275 | 0.2802 |
| mlm | concert_k48 | 1 | 0.2689 | 0.1710 | 0.3389 | 0.2878 |

Main paired comparisons (reference base_k16):
- **concert_k16, ssgsea:** raw +0.0097 (39/57, p = 0.016; groups 10/16, p = 0.13); depth-res +0.0050 (35/57, p = 0.11; groups 13/16, p = 0.029). READ: raw +0.022, depth-res +0.006.
- **concert_k16, mlm:** raw +0.0062 (30/57, p = 0.30; groups 6/16, p = 0.43); depth-res +0.0025 (p = 0.36; groups 5/16, p = 0.23). READ: raw −0.030, depth-res −0.015.
- **mlp (non-spatial):** ssgsea raw +0.0109 (p = 0.23), depth-res −0.0060 (p = 0.75; groups 5/16, p = 0.093), not distinguishable from the distance baseline. Clearly worse on mlm: raw −0.0118 (19/57, p = 0.0024; groups 3/16, p = 0.0076), depth-res −0.0098 (p = 0.0036; groups 2/16, p = 0.0027).
- **Ablations (reference concert_k16, ssgsea):** without H&E similarity, depth-res −0.0126 (p = 1.3e-4; groups 4/16, p = 0.021); without compartments −0.0147 (p = 6.1e-4; groups 4/16, p = 0.016); without distance −0.0053 (p = 0.27). noCos and noComp are also below base_k16 itself (−0.0076, p = 0.047; −0.0098, p = 0.055), so the three terms are not simply additive. On mlm, no ablation differs from concert_k16.
- **k = 48 (seed 0 only, paired with base_k16 seed 0):** ssgsea base_k48 raw +0.0024 (p = 0.97), depth-res −0.0031 (p = 0.66); concert_k48 vs concert_k16 depth-res −0.0113 (p = 0.085). On mlm, concert_k48 vs concert_k16 depth-res −0.0042 (p = 0.040).
- **Seed noise** (`B1_per_seed_effects.csv`): base_k16 seed 0 vs seed 1, ssgsea depth-res −0.0196 (15/57, p = 0.0013); mean absolute slide difference 0.059 raw and 0.036 depth-res; mlm depth-res −0.0099 (p = 2e-4). Concert vs base by seed: ssgsea raw +0.0107 / +0.0088 (same sign), depth-res +0.0115 / −0.0016 (opposite signs). **Slide-level p-values do not include training-seed variance; differences of order 0.01 in B1 should not be read as method differences.**

Figure: `fig_B1_spatial_ablation.png` (points = patient-group means, orange = READ, diamonds = mean over 57 slides ± 95% CI).

## 3. B2 — Prediction heads
- **Backbone:** concert_k16 (when B2 started, 7/8 ssgsea tier-1 runs were complete and concert_k16 was the only leading variant; the later mlm results showed concert ≈ base).
- (a) **det:** B1 concert_k16 predictions reused (two seeds), not retrained.
- (b) **gauss:** mean + low-rank (rank 4) + diagonal covariance Σ = FFᵀ + diag(d); loss = exact multivariate Gaussian NLL (`LowRankMultivariateNormal`, divided by P) + the baseline loss on the mean.
- (c) **flow:** conditional rectified flow. The velocity MLP takes [x_t (36), sinusoidal time encoding (32), projected backbone embedding (512)], architecture 512→512→512→36 with SiLU. Source distribution N(0, I) in robust-normalised space, path x_t = (1 − t)·x0 + t·y, target y − x0. Inference: 32 Euler steps, 16 samples per spot; the point prediction is the sample mean. Loss = flow-matching loss + baseline loss of an auxiliary mean head. Early stopping uses validation Pearson of the point prediction from 8 samples × 16 steps.
- Training as in B1 (30 epochs, bs = 1024, lr = 2e-4, bf16), 2 seeds × 4 folds × 2 targets.
- Metrics (per slide):
  - raw / depth-res Pearson.
  - ‖C_pred − C_true‖_F for the 36×36 inter-program correlation matrix, computed for the point prediction and for a single sample.
  - S matrix: S(g1, g2) = Spearman(g1 at a spot, neighbour mean of g2), with neighbours within 1.5× the median nearest-neighbour distance, i.e. the 6 hexagonal Visium neighbours rather than 8-connectivity (Visium is not a square grid). ‖S_pred − S_true‖_F and the Pearson correlation of the flattened matrices are reported.
  - Boundary Pearson, two definitions, labels computed once per slide at analysis time: (i) k-means (k = 6) on z-scored true programs, a spot is a boundary spot if any neighbour has a different label — this marks 88% (ssgsea) and 84% (mlm) of spots; (ii) smoothed: k-means on the mean of each spot and its 6 neighbours, giving 64% / 51% boundary spots; the figure uses (ii).
  - Calibration: coverage of central 50% / 90% intervals (Gaussian: analytic and 16-sample empirical quantiles; flow: 16 samples only), summarised by patient group; CRPS and energy score in robust-normalised units (det CRPS = MAE as reference); spread–skill = Spearman(sample SD, |error|).

### 3.1 Results (57 slides; `B2_head_summary.csv`, `B2_paired_tests.csv`, `B2_calibration_summary.csv`, `B2_calibration_by_patient_group.csv`)
| Target | Head | raw | depth-res | boundary (smoothed) | ‖ΔC‖ point | ‖ΔC‖ sample | S r point | S r sample | CRPS | energy score | cov50 | cov90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ssgsea | det | 0.372 | 0.162 | 0.328 | 11.43 | – | 0.668 | – | 0.567 | 3.95 | – | – |
| ssgsea | gauss | 0.365 | 0.166 | 0.324 | 11.41 | 4.96 | 0.674 | 0.793 | 0.439 | 3.01 | 0.355 (analytic) / 0.319 (16 samples) | 0.715 / 0.628 |
| ssgsea | flow | 0.334 | 0.139 | 0.294 | 9.26 | 5.56 | 0.691 | 0.700 | 0.459 | 3.19 | 0.271 | 0.557 |
| mlm | det | 0.271 | 0.176 | 0.233 | 14.78 | – | 0.632 | – | 0.513 | 3.88 | – | – |
| mlm | gauss | 0.265 | 0.171 | 0.229 | 15.23 | 5.19 | 0.634 | 0.677 | 0.377 | 2.84 | 0.409 / 0.366 | 0.798 / 0.704 |
| mlm | flow | 0.228 | 0.139 | 0.194 | 11.43 | 5.44 | 0.642 | 0.653 | 0.390 | 2.94 | 0.330 | 0.653 |

- The mean off-diagonal correlation between true programs is 0.528 (ssgsea) and 0.090 (mlm). Point predictions inflate it to 0.738 / 0.215 (det), i.e. point predictions are pulled towards the shared component. Single samples are closer to the truth (gauss 0.529 / 0.105, flow 0.551 / 0.101).
- gauss vs det: no significant change in accuracy (ssgsea raw −0.0073, p = 0.15, but only 2/16 groups improve, p = 0.025; mlm depth-res −0.0047, p = 0.0038). CRPS −0.128 / −0.136, energy score −0.94 / −1.03 (all p < 1e-9; ≤ 1/16 groups worse).
- flow vs det: raw −0.0376 / −0.0430, depth-res −0.0228 / −0.0368, smoothed boundary −0.034 / −0.039 (all p < 1e-6; 0–2/16 groups improve). Point-prediction ‖ΔC‖ is lower (−2.17 / −3.35).
- flow vs gauss (samples): ‖ΔC‖ sample +0.60 (ssgsea, p = 1.6e-4; groups p = 0.051) / +0.25 (mlm, p = 1.4e-3; groups p = 0.21); S r sample −0.093 / −0.024 (groups p = 0.025 / 0.011); CRPS +0.020 (p = 0.078) / +0.012 (p = 1.2e-3; groups p = 0.19). Flow does not beat Gaussian samples on any metric.
- Patient-group calibration: Gaussian analytic 90% coverage, ssgsea mean 0.695 (0.361–0.897, all 16 groups under-covered), mlm 0.771 (0.648–0.880); flow 90% coverage, ssgsea 0.563 (0.349–0.784), mlm 0.631 (0.507–0.733). Spread–skill is only 0.02–0.05, so per-spot uncertainty barely predicts error size.
- READ: on ssgsea, gauss raw 0.338 and flow 0.385 are both below det (0.457); on mlm, gauss 0.334 ≈ det 0.335.

Figure: `fig_B2_heads.png`.

## 4. Deviations from the planned protocol
1. bs = 1024 / lr = 2e-4 (square-root scaling) with bf16 autocast instead of bs = 256 / lr = 1e-4 / fp32, because the GPU was shared with two other jobs: at bs = 256 a run took 20–40 min and memory overflow caused severe slow-downs. All variants, including the baseline, use the same setting. Baseline fold 0 gave 0.237 (reference bundle), 0.343 (bs 256, bf16) and 0.303 (bs 1024) under the three settings, so within-fold variance exceeds the effect of the setting.
2. The two k = 48 variants were run with seed 0 only (16 runs; 2 seeds were planned).
3. Both Gaussian and flow heads include the baseline loss on the mean; pure-NLL / pure flow-matching controls were not run.
4. The det head reuses B1 predictions. The B2 backbone was chosen from interim ssgsea results, before the mlm results were available.
5. Compartment k-means is fitted on all non-test slides of each fold (training + inner validation group), never on test slides, in PCA-32 space.
6. The S matrix uses the 6 hexagonal neighbours instead of 8-connectivity; a smoothed boundary definition was added; flow intervals can only be estimated from 16 samples, which is known to under-estimate coverage (for the same Gaussian distribution, 16-sample 90% coverage is 0.628 vs 0.715 analytic).
7. Because of GPU contention some runs were stopped and restarted; every stopped run was re-run from scratch, so results are unaffected. No multiple-comparison correction.

## 5. Files
Code: `spatial_models.py`, `run_b1.py`, `run_b2.py`, `run_queue.py`, `b2_det.py`, `analyze_b1.py`, `analyze_b2.py`. Per-slide × program tables: `results/tables/stage1_hestbench/B1_*.csv`, `B2_*.csv`.
