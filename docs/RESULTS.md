# Results

> **Historical record (stages 1–2, H-optimus-0, 18 patient groups).** The current summary, with H-optimus-1 and 16 groups (the 2 groups with linked training slides excluded), plus the stage-3 headroom, composition and MDACC analyses, is [`CORE_FINDINGS.md`](CORE_FINDINGS.md). Conclusions here were not changed by the encoder update (CORE_FINDINGS section 3).

Every value here is read from `results/tables/`. The analysis scripts in `scripts/07_analysis/` regenerate those tables from `results/raw/`.

**How to read the numbers:**
- *Group-level* means are averaged over CRC patient groups: 18 in stage 2, 16 in stage 1.
- *Depth-res* is the depth-residualised Pearson r, the primary metric.
- Paired tests are two-sided Wilcoxon signed-rank tests over patient groups. *k/n* is the number of groups favouring the first condition.
- Seed-noise SD of the group-level difference: ssGSEA 0.036, ULM 0.027.

## Stage 1: HEST-Bench (57 Visium WT slides, 16 patient groups)

Tables are in `results/tables/stage1_hestbench/` and figures in `results/figures/stage1_hestbench/`.

| Question | Result |
|---|---|
| Shared "busy-spot" axis (A1, median PC1 variance fraction of the 36 program maps) | ssGSEA 0.60 vs random sets 0.52; ULM 0.35 vs 0.13; MLM 0.22 vs 0.11. ssGSEA maps are dominated by one shared axis. |
| Split-half reliability, depth-res (A2) | ssGSEA 0.28 (cross-program 0.11); ULM 0.25 (cross-program 0.06); MLM 0.13 (cross-program 0.03). ULM gives the best program-specific agreement. |
| Held-out CV, slide mean (A3/A4) | Raw: ssGSEA 0.34 vs random 0.31–0.32; ULM 0.35 vs 0.15. Depth-res: ssGSEA 0.15 vs about 0.08; ULM 0.23 vs 0.05; ssGSEA depth-regressed 0.23. |
| Spatial model (B1) | No CONCERT-bias, ablation or k = 48 variant beats distance-only SPARC-Map k = 16 by more than seed noise. Centre-only MLP is slightly worse. |
| Probabilistic heads (B2) | Gaussian head matches deterministic accuracy (ssGSEA raw 0.365 vs 0.372). Flow head is lower (0.334). Gaussian samples reproduce inter-program correlation better (‖ΔC‖_F 4.96 vs 11.4 for deterministic). Raw 90% intervals under-cover (0.69 ssGSEA, 0.77 MLM), so group-level conformal recalibration is needed. |
| Active learning (slide budget 2 → 14) | Batch EPIG beats random (AUC 0.345 vs 0.287; 12/12 fold × seed, p = 5e-4) and max-variance (0.293). It does **not** beat the model-free H&E-embedding-match baseline (0.354). |

## Stage 2: HEST-1k → 58 CRC slides (18 patient groups, 6-fold)

### 2.1 Targets are program-specific only after removing shared and depth components

Source: `hest1k_condition_summary.csv`, `audit_leakage_sensitivity.csv`. Figures: `fig_hest1k_real_vs_random.png`, `fig_hest1k_targets.png`.

| Target (trained on HEST-1k) | Raw | Centred | Depth-res | Random-set depth-res |
|---|---|---|---|---|
| ssGSEA | 0.458 | 0.412 | 0.234 (slide) / 0.246 (group) | 0.06 (3 draws) |
| ULM | 0.452 | 0.431 | 0.329 (slide) / 0.346 (group) | 0.09 |
| MLM | 0.386 | 0.372 | 0.260 | – |

- Raw ssGSEA reaches about 90% of its accuracy with random gene sets (group level 0.45 vs 0.40). Raw accuracy therefore mostly reflects predictable shared structure.
- After depth residualisation, real programs beat random sets: ssGSEA +0.18 and ULM +0.25 (18/18, p = 7.6e-6). All 36 ULM programs exceed their random counterpart (`hest1k_real_vs_random_per_program.csv`).
- Target comparisons at the group level, depth-res (`hest1k_wave1_paired_tests.csv`):
  - ULM − ssGSEA: +0.100 (17/18, p = 1.5e-5).
  - ULM − MLM: +0.073 (18/18).
  - MLM − ssGSEA: +0.027 (13/18, p = 0.08).
  - ULM − depth-regressed ssGSEA: +0.010 (12/18, p = 0.25).

### 2.2 Training scale

- HEST-1k pool (about 376 slides/fold) vs CRC-only (about 48 slides/fold), depth-res:
  - ULM +0.019 (14/18, p = 0.04).
  - ssGSEA +0.027 (15/18, p = 8e-4).
  - MLM +0.016 (n.s.).
- This is a small but consistent gain. After excluding the two leaked groups, the ULM gain is +0.020 (12/16, p = 0.07); see `AUDIT.md`.

### 2.3 Spatial modelling and heads

Figure: `fig_hest1k_spatial.png`. All differences are depth-res vs base SPARC-Map.

| Variant | ssGSEA | ULM |
|---|---|---|
| Centre-only MLP (no context) | −0.021 (2/18, p = 0.005) | −0.021 (2/18, p = 4e-5) |
| CONCERT-style bias | +0.005 (p = 0.90) | −0.003 (p = 0.8) |
| GP kernel k = 16 | −0.012 (p = 0.06) | +0.002 (p = 0.77) |
| GP kernel k = 48 | −0.005 (p = 0.26) | +0.002 (p = 0.70) |
| Gaussian head | +0.001 (p = 0.90) | 0.326 vs 0.329 (slide mean, 2 seeds; not tested) |

Spatial context helps (about 0.02), but no richer spatial prior beats SPARC-Map's distance-only neighbour attention.

### 2.4 Proposed upgrades (wave 2, ULM, depth-res vs base)

Source: `hest1k_wave2_paired_tests.csv`. Figure: `fig_upgrades.png`.

| Upgrade | Δ | Groups | p |
|---|---|---|---|
| Slide z-score targets | −0.024 | 3/18 | 7e-4 |
| Technology one-hot | −0.003 | 9/18 | 0.73 |
| Program reliability weights | −0.003 | 8/18 | 0.25 |
| All three combined | −0.021 | 5/18 | 0.002 |
| Gene auxiliary loss | +0.001 | 9/18 | 0.90 |
| Gene decoder + ULM readout (2 seeds) | +0.011 (raw −0.063) | 13/18 | 0.14 |
| 448 µm context, HEST-1k | −0.009 | 7/18 | 0.26 |
| 448 µm context, CRC-only | −0.014 | 4/18 | 0.01 |

The base model reproduced across waves (wave-2 base 0.3465 vs wave-1 0.3461).

### 2.5 Composition decoupling

Source: `composition_r2_by_program.csv`, `decoupling_residual_per_program.csv`. Figures: `composition_r2_by_program.png`, `fig_decoup.png`.

- 12-type NNLS composition explains a within-slide median R² of 0.25 of ULM program variance. It explains most for EMT (0.64), ECM organisation (0.60) and integrin interactions, and least for macrophage M1-vs-M2 (DN, 0.07), PD1-high CD8 T-cell (0.07) and mismatch-repair (0.08) programs.
- Composition-residual targets remain predictable from H&E: 0.146 vs 0.032 for residualised random sets (18/18, p = 7.6e-6). Residual signal exceeds random in 34/36 programs, and is largest for stromal programs (EMT, ECM, collagen ≈ +0.30).
- Residual signal correlates with composition R² across programs (Spearman 0.45). See `AUDIT.md` for why this is an upper bound.

### 2.6 Nuclear morphology

Source: `nuclei_adjusted_eval.csv`. Figure: `fig_nuclei_adjust.png`. Scope: 45 FFPE slides, 16,143 sampled spots, 11 groups.

| Adjustment | ULM (1 seed) | 3-seed ensemble | Random | Δ vs random |
|---|---|---|---|---|
| Depth | 0.347 | 0.357 | 0.087 | +0.260 (11/11, p = 0.001) |
| Depth + nuclear features | 0.316 | 0.324 | 0.077 | +0.238 (11/11, p = 0.001) |

- 91% of the depth-adjusted ULM accuracy survives additional adjustment for nucleus count, density, PanNuke type fractions and nuclear size.
- Nuclear features explain a median 12% of ULM variance. Nuclear density correlates only weakly with log UMI (Spearman 0.25).
- HoVer-Net types agree moderately with deconvolution: neoplastic–Epi_tumor 0.39, connective–fibroblast 0.39, inflammatory–immune 0.12.
- The 3-seed ensemble adds +0.009 (p = 0.03).

### 2.7 Mouse

Source: `mouse_external_by_organ.csv`, `mouse_ft_summary.csv`, `mouse_ft_tests.csv`. Figure: `fig_mouse_ft.png`. Scope: 22 sections; slide-level means.

| Condition (ULM, depth-res) | Slide-scheme CV | Leave-dataset-out | Mouse random null (slide / dataset) |
|---|---|---|---|
| Human-only, applied to mouse | 0.102 | 0.102 | 0.031 |
| Mouse-only | 0.259 | 0.102 | 0.087 / 0.022 |
| Human → mouse fine-tune | 0.245 | 0.109 | 0.080 / 0.034 |

- **Human-only transfer** is weak: +0.071 over the mouse null (21/22). By organ: bowel 0.15, skin 0.09, hepatoblastoma liver 0.04.
- **Mouse retraining** helps, but only under the slide scheme: mouse-only vs human-only +0.157 (20/22, p = 3e-6). The gain over the null is +0.172. Fine-tuning from human is slightly worse than mouse-only (−0.014, 2/22, p = 0.004).
- **Under leave-dataset-out**, there is no gain: +0.000 (p = 0.85). The slide-scheme gain is therefore within-dataset. Without animal IDs, serial-section sharing cannot be excluded.

## Implications for model development

1. Use ULM targets, and always report against size-matched random gene sets and after depth, composition and nuclear adjustment.
2. Keep SPARC-Map's spatial encoder. The CONCERT GP prior moves to the perturbation module, not the H&E encoder.
3. Add a Gaussian head with conformal recalibration for uncertainty-aware maps and sampling.
4. Train on an in-domain mouse atlas, using animal-level splits.
5. Treat human pretraining as "to be tested".
6. For acquisition, EPIG must beat an H&E-similarity baseline before it is claimed.
