# Track A — Program targets (proposal Aim 2, claim i)

## 中文摘要
- **ssGSEA 目标的共享成分主要来自打分方式本身。** 36 个与真实程序等大小的随机基因集，用 ssGSEA 打分后 PC1 方差占比中位数为 0.520，程序间平均 r 为 0.471；真实程序分别为 0.598 和 0.537。用 ULM/MLM 打分时，随机集几乎没有共享结构（平均 r 0.081/0.059）。
- **原始 Pearson 不能作为程序生物学的证据。** 在相同 SPARC-Map 训练流程下，随机基因集的 ssGSEA 目标留患者外 Pearson 为 0.316（3 次抽样均值），真实程序为 0.342。患者组层面差异 +0.025，Wilcoxon p = 0.21（16 组）。只有中心化后（0.246 vs 0.044）或深度残差化后（0.147 vs 0.083），真实程序才明显优于随机。
- **联合估计（MLM）并未带来更可重复的程序特异信号。** 深度残差化后，MLM 的半集一致性最低（0.134），ssGSEA 为 0.277，ULM 为 0.249。程序特异性（同程序减跨程序）排序为：ULM 0.188 ≈ ssGSEA 中心化 0.191 > ssGSEA 0.162 > MLM 0.105。
- **H&E 可预测性。** 程序特异指标上 ULM 最好：中心化 0.290，深度残差化 0.233；MLM 为 0.233/0.177，ssGSEA 为 0.246/0.147。MLM 只在深度残差化指标上优于 ssGSEA（+0.030，p = 0.002），中心化指标上无差异（p = 0.18）。直接对 ssGSEA 做深度残差化，效果与 ULM 相当（0.233 vs 0.233，p = 0.91）。
- **Spectra（仅 8 张切片，暂定结论）。** 共享成分居中：PC1 0.440，ssGSEA 0.649，MLM 0.234；未优于 MLM。
- **对提案的建议：** 把“联合估计更好”改写为“需要显式去除深度/共享成分，并以随机基因集零模型为对照”。主指标应改为中心化/深度残差化 Pearson 加随机零模型差值，ULM 与 ssGSEA 深度残差化作为强基线。

## Data and splits
- 57 HEST-Bench Visium whole-transcriptome slides (CCRCC 24, PRAD 23, LYMPH_IDC 4, READ 4, HCC 2) in 16 patient groups. 4-fold patient-group CV from `sparcmap_lib.wt_folds()`. The 15 Xenium slides were excluded, as instructed.
- H-optimus-0 patch features (1536-d). Targets come from `targets/<sample>.parquet` (ssGSEA / ULM / MLM over 36 SPARC programs, MSigDB 2026.2.Hs).
- Raw counts: `MahmoodLab/hest-bench` adata, used for A2, A3 and Spectra.

## A1 — Shared components
- Computed per slide and target type: PC1 variance fraction of the 36 z-scored program maps; mean off-diagonal inter-program r; mean |r| of each program with log(total counts); and |r| of PC1 with log depth.
- Target types: ssgsea, ulm, mlm; ssgsea_depthres (OLS residual on log1p total counts, within slide); ssgsea_centred (per-slide z-score, then the per-spot mean across programs subtracted).
- The same statistics were computed for the A3 random gene sets (3 draws, all three scorings).
- Spots with NaN ULM/MLM (zero-count spots; ≤61 per slide, 8 slides) were dropped.
- **Spectra** (scSpectra 0.2.1, CPU):
  - Global model (`use_cell_types=False`), L = 41 (36 factors anchored to the programs + 5 free), lam = 0.01, 2,000 epochs (default is 10,000), seed 0.
  - Input: log1p(CP10k). Gene panel: program genes detected in ≥5% of spots plus 500 HVGs (2,895–3,809 genes).
  - Factor j < 36 is the factor initialised from program j. Its top-50-gene overlap coefficient with the anchor set had a median of 1.0.
  - Slides: ZEN36, ZEN40, ZEN48, ZEN49 (READ); INT1, INT10 (CCRCC); MEND147, MEND161 (PRAD). Runtime 403–767 s per slide.
  - `pkg_resources` imports were patched to optional, because setuptools 84 no longer provides it.

## A2 — Split-half reliability
- For each program, the nominal gene list was split randomly into two halves, 10 times (rng seed 7). Each half was intersected with the genes detected on the slide.
- Scorings:
  - ssGSEA: closed form identical to `compute_targets.ssgsea_gpu`.
  - ULM/MLM: re-implemented in torch (t-values of OLS on log1p CP10k with intercept). They match the stored decoupler-2.2 values at r = 1.0000 for all 36 programs on 3 test slides (max abs diff < 1e-6).
  - MLM halves were fitted jointly with the other 35 programs' same-half sets.
  - ssgsea_centred halves were centred within each half-matrix.
- Metrics:
  - Within-slide Pearson between half A and half B, raw and after residualising both halves on log depth.
  - Added metric, **specificity** = same-program depth-residualised split-half r minus the mean cross-program r (half A of program i vs half B of programs j ≠ i).
- Aggregation: averaged over repeats, then over programs per slide. Wilcoxon signed-rank tests on 57 slides and on 16 patient-group means.

## A3 — Random-gene-set null
- Universe: 13,586 genes detected in ≥90% of the 57 WT slides.
- 36 random sets per draw, each the size of the corresponding program's overlap with the universe (9–400 genes). Draws were made without expression matching (rng seed 2026; `random_gene_sets.json`).
- All 3 draws were scored with ssGSEA, ULM and MLM. The scorer code and settings were the same as for the real targets.
- H&E models trained on the random targets: ssGSEA draws 0, 1, 2; MLM draw 0; ULM draw 0 (the ULM run goes beyond the task).

## A4 — H&E predictability
- Model: baseline SPARC-Map via `sparcmap_lib.train` (k = 16 neighbours, H = 256, 3 blocks, 8 heads, MSE + 0.5·(1 − Pearson), AdamW lr 1e-4, wd 1e-2, bs 256, robust label normalisation), 30 epochs.
- Early stopping on an inner held-out patient group, chosen exactly as in `run_baseline.py`.
- Training seed = fold index (identical to the baseline). **One seed for every target.**
- The real-ssGSEA result is the baseline bundle's `baseline_sparcmap_ssgsea_cv.csv`; it was not retrained.
- Metrics from `L.evaluate`: per-slide, per-program raw Pearson; centred Pearson (per-spot mean across z-scored programs removed from both prediction and truth); depth-residualised Pearson.
- Summaries: per-slide means over programs, then the mean over 57 slides, the mean of 16 group means, and READ separately (4 slides, 2 groups).

## Results (copied from saved tables)
| target | n slides | Pearson | centred | depth-res. | READ Pearson | READ centred | READ depth-res. |
|---|---|---|---|---|---|---|---|
| ssGSEA (real) | 57 | 0.342 | 0.246 | 0.147 | 0.456 | 0.323 | 0.248 |
| ssGSEA random (mean of 3 draws) | 57 | 0.316 | 0.044 | 0.083 | 0.365 | 0.055 | 0.026 |
| ULM (real) | 57 | 0.347 | 0.290 | 0.233 | 0.416 | 0.364 | 0.375 |
| ULM random (1 draw) | 57 | 0.147 | 0.081 | 0.051 | 0.163 | 0.120 | 0.066 |
| MLM (real) | 57 | 0.259 | 0.233 | 0.177 | 0.354 | 0.329 | 0.291 |
| MLM random (1 draw) | 57 | 0.143 | 0.086 | 0.055 | 0.168 | 0.131 | 0.087 |
| ssGSEA depth-residualised target | 57 | 0.221 | 0.209 | 0.233 | 0.331 | 0.294 | 0.342 |

Real − random gap (Wilcoxon on slides n = 57 / on groups n = 16):
- ssGSEA raw +0.027 (p = 0.015 / 0.21); centred +0.202 (5.4e-11 / 3.1e-05); depth-res. +0.064 (9.3e-07 / 0.0021).
- ULM raw +0.199; centred +0.209; depth-res. +0.182 (all p ≤ 5.4e-11 / 3.1e-05).
- MLM raw +0.116; centred +0.146; depth-res. +0.122 (p ≤ 3e-10 / 3.1e-05).

Paired comparisons of target types:
- MLM − ssGSEA: raw −0.083 (p = 1.6e-06 / 5.8e-04); centred −0.014 (0.18 / 0.82); depth-res. +0.030 (0.002 / 0.0092).
- ULM − ssGSEA: raw +0.004 (0.67 / 0.90); centred +0.044 (4.3e-07 / 0.001); depth-res. +0.086 (5.9e-09 / 6.1e-05).
- ULM − MLM: depth-res. +0.056 (3e-10 / 3.1e-05).
- ULM − ssGSEA-depthres: depth-res. +0.000 (p = 0.91 / 0.32).

READ: number of programs (of 36) with depth-residualised Pearson > 0.3 — ssGSEA 12, ULM 23, MLM 16, ssGSEA-depthres 21; 0 for every random-set target.

A1 (median over 57 slides), PC1 fraction / mean inter-program r:

| scoring | real programs | random sets |
|---|---|---|
| ssGSEA | 0.598 / 0.537 | 0.520 / 0.471 |
| ULM | 0.351 / 0.255 | 0.130 / 0.081 |
| MLM | 0.225 / 0.078 | 0.111 / 0.059 |
| ssGSEA centred | 0.230 / −0.015 | — |
| ssGSEA depth-res. | 0.232 / 0.163 | — |

- Under ssGSEA, each random-set score correlates on average r = 0.488 with its size-matched real program; ULM 0.156, MLM 0.095.
- Spectra, 8 slides (provisional), compared on the same slides: PC1 fraction 0.440 vs ssGSEA 0.649, ULM 0.399, MLM 0.234. Mean per-program |r| with depth: 0.562 vs MLM 0.390 (means over the 8 slides).

A2 (mean over 57 slides), depth-residualised split-half r / specificity:

| scoring | split-half r | specificity |
|---|---|---|
| ssGSEA | 0.277 | 0.162 |
| ssGSEA centred | 0.189 | 0.191 |
| ULM | 0.249 | 0.188 |
| MLM | 0.134 | 0.105 |

- Specificity differences: ULM − ssGSEA +0.026 (p = 9.1e-05 slides / 0.001 groups); MLM − ssGSEA −0.056 (2.4e-09 / 5.8e-04); MLM − ULM −0.082.

## Deviations from the task
See the `deviations` list in the structured output. In short:
1. One training seed for every target. The optional 3-seed ssGSEA-vs-MLM comparison was not run.
2. The MLM random null has 1 trained draw, not 3. Score-level null statistics use all 3 draws.
3. The real-ssGSEA CV result is reused from the baseline bundle.
4. Spectra was run on 8 slides only, with reduced epochs and a reduced gene panel, and was not trained on H&E.
5. ULM/MLM for A2/A3 used a validated torch re-implementation.
6. Zero-count spots were dropped.
7. The random sets were size-matched, not expression-matched.
8. Up to 3 concurrent training processes were run (≤2.5 GB peak reserved each; ~7.5 GB total at the peak).

## Files
- Scripts: `score_lib.py`, `score_A2_A3.py`, `run_cv.py`, `spectra_subset.py`, `analyse_cv.py`.
- Tables:
  - A1: `A1_shared_components_per_slide.csv`, `A1_shared_components_summary.csv`, `A1_spectra_subset_per_slide.csv`.
  - A2: `A2_splithalf_per_slide.csv`, `A2_splithalf_summary.csv`, `A2_splithalf_paired_tests.csv`, `A2_splithalf_per_program.csv`.
  - A3: `A3_real_vs_random_score_similarity_per_slide.csv`.
  - A3/A4: `A34_cv_per_slide.csv`, `A34_cv_summary.csv`, `A34_cv_by_task.csv`, `A34_paired_tests.csv`, `A4_READ_per_program.csv`.
- Figure: `trackA_targets_figure.png` / `.pdf`.
