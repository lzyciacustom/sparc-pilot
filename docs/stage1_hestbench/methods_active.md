# Track C: active selection of sections (proposal Aim 2, claim iii)

## What was run
* **Data:** the 57 HEST-Bench Visium whole-transcriptome slides, with ssGSEA targets for the 36 SPARC programs and H-optimus-0 patch features (1536-d). The folds are the 4 patient-group CV folds from `L.wt_folds()`. **MLM targets were not run** because of compute time (see Deviations).
* **Per fold k:**
  * Target distribution = the fold-k test slides. Their H&E features are used unlabelled for EPIG and featmatch scoring; their targets are used only for scoring the learning curves.
  * Inner early-stopping set = one patient group taken from the training folds, chosen exactly as in `run_baseline.py`: the median-sized group, which is CCRCC_f1 for folds 0, 1 and 3 and LYMPH_IDC_f3 for fold 2. This group is labelled, fixed, identical for all strategies, and **excluded from the acquisition pool**.
  * Pool = the remaining training-fold slides (36/38/42/42 slides for folds 0–3).
* **Protocol:**
  * Start with 2 random pool slides, drawn from 3 random initial sets per fold (seeds 0,1,2).
  * Acquire 2 slides per round for 6 rounds, giving 2→14 labelled slides and 7 evaluation points.
  * Within a (fold, seed), the initial set and the round-0 ensemble are shared by all strategies.
  * 4 folds × 3 seeds = **12 paired runs per strategy**.
* **Model and training (cheap schedule):**
  * Architecture and loss: `L.build_model` (SPARC-Map: k=16 neighbours, relative-coordinate MLP, 3-layer transformer, H=256, 8 heads) with MSE + 0.5·(1−Pearson).
  * Optimisation: robust label normalisation; AdamW lr 1e-4, wd 1e-2; batch size 256; gradient clipping 1.0.
  * Schedule: **8 epochs** (baseline used 30). Steps/epoch = max(50, labelled spots/256). Early stopping on the inner validation group (per-slide mean Pearson).
  * Implemented in `al_core.py`, a re-implementation of `L.train`: features stored fp16 on the GPU, bf16 autocast.
* **Ensemble:**
  * M = 5 members per round, with different init/batch seeds.
  * Each member also draws its slide-sampling weights from a Dirichlet(1) Bayesian bootstrap over the labelled slides.
  * Evaluation uses the **ensemble-mean** prediction for every strategy, including random.
  * Total: 48 strategy runs × 7 rounds, i.e. 1,500 member trainings.
* **Metrics:**
  * Per test slide: mean over the 36 programs of the raw Pearson and the depth-residualised Pearson. Residualisation regresses both prediction and target on log1p(total counts) within the slide.
  * Per run: the mean over the fold's test slides.
  * AUC = trapezoidal area over rounds 0–6, divided by 6, i.e. the mean accuracy along the curve.
  * **Spot-budget variant (post hoc):** the same runs re-indexed by cumulative labelled spots. Each (fold, seed) curve is interpolated onto a common grid from the initial spots to the minimum final spot count over the 4 strategies, and the normalised AUC is computed on that grid. No cost-aware selection was run.
* **Statistics:**
  * Paired Wilcoxon signed-rank tests at three levels: (a) the 12 fold×seed runs; (b) the 57 test slides, using per-slide AUC averaged over seeds; (c) the 16 patient groups, using the mean of the slide AUCs within each group.
  * READ (4 test slides, 2 groups) is reported descriptively, without a test.

## Acquisition rules (exact)
All scores are computed with the current round's ensemble, in the common robust-normalised units of member 0.
1. **random**: 2 slides uniformly without replacement from the unlabelled pool.
2. **maxunc**: for each candidate slide, the variance across the 5 members of each program's prediction is summed over the 36 programs and averaged over **all** spots of the slide. The top-2 slides are taken, with no diversity term.
3. **epig**: for each candidate slide, 300 spots are sampled; for the targets, 100 spots are sampled from each test slide.
   * For each program, the members' predictions are centred across members to give 5-vectors. ρ(x,x*) is the cosine between those vectors at x and x*, which equals the correlation across members.
   * EPIG(slide) = mean over all (x, x*) pairs and the 36 programs of −0.5·log(1−ρ²), with ρ² clipped at 0.999. This is the noise-free joint-Gaussian form specified in the task.
   * **Batch rule:** the first pick is argmax EPIG. The second pick is argmax EPIG(s)·(1 − max_(c∈chosen) R(s,c)), where R(s,c) is the mean over programs and sampled spot pairs of ρ² between the member-deviation vectors of s and c. This discounts candidates whose ensemble disagreement duplicates a slide already chosen.
4. **featmatch** (extra control, not in the task): a model-free and label-free rule. Each candidate's score is its mean cosine similarity between the slide-mean H-optimus-0 embedding and each target slide's mean embedding. The top-2 slides are taken, with no diversity term.

## Results (ssGSEA targets, 12 paired runs per strategy, mean ± SEM over runs)
| strategy | AUC raw r | AUC depth-res. r | final (14 slides) raw | final depth-res. | spot-AUC raw | spot-AUC depth-res. |
|---|---|---|---|---|---|---|
| random | 0.287 ± 0.016 | 0.117 ± 0.015 | 0.341 | 0.138 | 0.277 | 0.114 |
| maxunc | 0.293 ± 0.011 | 0.114 ± 0.014 | 0.344 | 0.136 | 0.285 | 0.112 |
| epig | 0.345 ± 0.008 | 0.135 ± 0.016 | 0.395 | 0.162 | 0.339 | 0.133 |
| featmatch | 0.354 ± 0.007 | 0.138 ± 0.019 | 0.405 | 0.165 | 0.346 | 0.135 |

Paired comparisons of AUC (A − B):

| comparison | metric | level | n | mean diff | n positive | Wilcoxon p |
|---|---|---|---|---|---|---|
| epig-random | auc_pearson | fold x seed run | 12 | +0.0580 | 12/12 | 0.0005 |
| epig-random | auc_pearson | test slide (seed-avg) | 57 | +0.0553 | 45/57 | <1e-4 |
| epig-random | auc_pearson | patient group (16) | 16 | +0.0867 | 14/16 | 0.0002 |
| epig-random | auc_pearson | READ slides (seed-avg) | 4 | +0.2149 | 4/4 | n/a |
| epig-random | auc_pearson_depthres | fold x seed run | 12 | +0.0178 | 11/12 | 0.0161 |
| epig-random | auc_pearson_depthres | test slide (seed-avg) | 57 | +0.0171 | 42/57 | 0.0005 |
| epig-random | auc_pearson_depthres | patient group (16) | 16 | +0.0233 | 12/16 | 0.0076 |
| epig-random | auc_pearson_depthres | READ slides (seed-avg) | 4 | +0.0532 | 3/4 | n/a |
| epig-maxunc | auc_pearson | fold x seed run | 12 | +0.0525 | 12/12 | 0.0005 |
| epig-maxunc | auc_pearson | test slide (seed-avg) | 57 | +0.0528 | 50/57 | <1e-4 |
| epig-maxunc | auc_pearson | patient group (16) | 16 | +0.0420 | 16/16 | <1e-4 |
| epig-maxunc | auc_pearson | READ slides (seed-avg) | 4 | +0.0260 | 3/4 | n/a |
| epig-maxunc | auc_pearson_depthres | fold x seed run | 12 | +0.0212 | 10/12 | 0.0161 |
| epig-maxunc | auc_pearson_depthres | test slide (seed-avg) | 57 | +0.0223 | 44/57 | 0.0001 |
| epig-maxunc | auc_pearson_depthres | patient group (16) | 16 | +0.0117 | 12/16 | 0.1591 |
| epig-maxunc | auc_pearson_depthres | READ slides (seed-avg) | 4 | +0.0028 | 2/4 | n/a |
| maxunc-random | auc_pearson | fold x seed run | 12 | +0.0055 | 5/12 | 0.9697 |
| maxunc-random | auc_pearson | test slide (seed-avg) | 57 | +0.0025 | 27/57 | 0.9398 |
| maxunc-random | auc_pearson | patient group (16) | 16 | +0.0446 | 11/16 | 0.0577 |
| maxunc-random | auc_pearson | READ slides (seed-avg) | 4 | +0.1890 | 4/4 | n/a |
| maxunc-random | auc_pearson_depthres | fold x seed run | 12 | -0.0034 | 7/12 | 0.7910 |
| maxunc-random | auc_pearson_depthres | test slide (seed-avg) | 57 | -0.0052 | 26/57 | 0.3111 |
| maxunc-random | auc_pearson_depthres | patient group (16) | 16 | +0.0116 | 12/16 | 0.1046 |
| maxunc-random | auc_pearson_depthres | READ slides (seed-avg) | 4 | +0.0504 | 4/4 | n/a |
| epig-featmatch | auc_pearson | fold x seed run | 12 | -0.0087 | 5/12 | 0.1514 |
| epig-featmatch | auc_pearson | test slide (seed-avg) | 57 | -0.0075 | 24/57 | 0.2781 |
| epig-featmatch | auc_pearson | patient group (16) | 16 | +0.0026 | 8/16 | 0.9399 |
| epig-featmatch | auc_pearson | READ slides (seed-avg) | 4 | -0.0332 | 1/4 | n/a |
| epig-featmatch | auc_pearson_depthres | fold x seed run | 12 | -0.0031 | 4/12 | 0.7334 |
| epig-featmatch | auc_pearson_depthres | test slide (seed-avg) | 57 | -0.0032 | 21/57 | 0.1632 |
| epig-featmatch | auc_pearson_depthres | patient group (16) | 16 | +0.0031 | 7/16 | 0.9399 |
| epig-featmatch | auc_pearson_depthres | READ slides (seed-avg) | 4 | +0.0040 | 3/4 | n/a |
| featmatch-random | auc_pearson | fold x seed run | 12 | +0.0667 | 11/12 | 0.0010 |
| featmatch-random | auc_pearson | test slide (seed-avg) | 57 | +0.0628 | 42/57 | <1e-4 |
| featmatch-random | auc_pearson | patient group (16) | 16 | +0.0841 | 13/16 | 0.0004 |
| featmatch-random | auc_pearson | READ slides (seed-avg) | 4 | +0.2481 | 4/4 | n/a |
| featmatch-random | auc_pearson_depthres | fold x seed run | 12 | +0.0209 | 10/12 | 0.0342 |
| featmatch-random | auc_pearson_depthres | test slide (seed-avg) | 57 | +0.0203 | 41/57 | 0.0010 |
| featmatch-random | auc_pearson_depthres | patient group (16) | 16 | +0.0202 | 10/16 | 0.1046 |
| featmatch-random | auc_pearson_depthres | READ slides (seed-avg) | 4 | +0.0493 | 3/4 | n/a |

Selection composition (all 144 acquired slides per strategy):

| strategy | frac. picks from an already-covered patient group | frac. batches with both picks from one group | distinct groups after 14 slides | frac. picks whose tissue type occurs in the target fold | mean spots per picked slide |
|---|---|---|---|---|---|
| random | 0.556 | 0.208 | 6.83 | 0.507 | 3028 |
| maxunc | 0.438 | 0.236 | 8.25 | 0.500 | 2738 |
| epig | 0.528 | 0.069 | 7.50 | 0.750 | 2977 |
| featmatch | 0.556 | 0.486 | 6.08 | 0.861 | 2759 |

The pool fraction of slides with a target-fold tissue type is 0.497 (mean over folds), so random sampling is expected to hit it about half the time.

Does the acquisition score predict the realised gain? This uses the random-strategy rounds only: the mean pool percentile of the 2 random picks is correlated with the change in test accuracy in that round (n = 72 rounds). The last row gives the median candidate-level Spearman correlation between the EPIG and maxunc scores.

| score | gain | n | Spearman | p |
|---|---|---|---|---|
| epig_pctile | gain_pearson | 72 | 0.272 | 0.021 |
| epig_pctile | gain_pdres | 72 | -0.012 | 0.917 |
| maxunc_pctile | gain_pearson | 72 | -0.000 | 0.998 |
| maxunc_pctile | gain_pdres | 72 | 0.117 | 0.329 |
| epig vs maxunc (candidate ranks) | - | 288 | 0.357 |  |

## Interpretation
* **EPIG beats random and max-uncertainty at equal slide budget**, and also at equal spot budget. The gain is large on raw Pearson (+0.058 AUC over random, 12/12 runs) and about 3× smaller on the depth-residualised metric (+0.018, 11/12 runs).
* **Max-uncertainty is no better than random**, at every level of analysis.
* **EPIG's advantage is explained by target-distribution matching.**
  * EPIG picks a target-fold tissue type 75% of the time, against 50% for random.
  * The model-free H&E-embedding-match control does at least as well as EPIG: EPIG − featmatch is −0.009 raw AUC (p=0.15) and −0.003 depth-residualised (p=0.73), with no difference at the 16-group level.
  * In this simulation, therefore, the ensemble-based mutual-information estimate adds nothing beyond knowing which pool slides look like the target H&E.
* **The redundancy penalty works as intended on near-duplicates.** Only 6.9% of EPIG batches take both slides from one patient group, against 20.8% (random), 23.6% (maxunc) and 48.6% (featmatch). This diversity did not translate into higher accuracy than featmatch.
* **About half of all acquisitions (44–56%) come from patient groups already in the labelled set**, under every strategy. Max-uncertainty covers the most groups.
* **Caveats:**
  * The target is the whole test fold. In folds 2–3 that fold is CCRCC/HCC/LYMPH_IDC, and the pool contains several same-organ groups, which makes distribution matching easy.
  * With M=5 the member-correlation estimate has only 4 degrees of freedom: its chance level is E[ρ²]=0.25, and observed EPIG values were only modestly above that chance level.
  * The noise-free EPIG ignores observation noise.

## Deviations from the task
* MLM targets were not run, because of compute time (~13 h of shared-GPU time for the ssGSEA design). All results are ssGSEA only.
* The spot-count budget is analysed post hoc on the slide-budget runs (re-indexed by spots). No cost-aware (score/spots) selection was simulated.
* The inner validation group is labelled, fixed per fold and excluded from the pool, so it is not part of the acquisition budget.
* The training schedule is 8 epochs (vs 30), with bf16 autocast and fp16 features. Ensemble diversity comes from seeds plus a Dirichlet slide-weight bootstrap.
* EPIG uses spot subsampling (300 candidate spots per slide, 100 target spots per test slide) and clips ρ² at 0.999.
* An additional control strategy (featmatch) was added. It was run after the main 3 strategies with the same seeds/initial sets.
* Number of seeds: 3 initial sets per fold × 5 ensemble members. The round-0 ensemble is shared within each (fold, seed).

## Files
Code: `al_core.py`, `run_al.py`, `analyze_al.py`, `make_figs.py`; raw per-run outputs: `al_raw_outputs_ssgsea.tar.gz`.
