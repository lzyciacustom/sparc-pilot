# sparc-pilot: null-calibrated evaluation of H&E → gene-program prediction

These are pilot experiments testing which upgrades to **SPARC** (H&E → spatial gene-program maps; Ayed et al., medRxiv 2026.05.04.26352410) are supported by data, and whether predicted program activity can be split into a cell-composition part and a residual (cell-state) part.

- **Data:** public data only (HEST-1k, GSE132465, GSE115469, COLON MAP, GSE294385).
- **Model and features:** a reproduction of the SPARC-Map spatial transformer on **H-optimus-1** features (H-optimus-0 available via `SPARC_ENCODER=h0`; earlier stage-2 results used H-optimus-0).
- **Test set:** 58 colorectal-cancer (CRC) slides; headline numbers use the 16 patient groups without linked training-pool slides (see `docs/AUDIT.md`).
- **Stage 1:** HEST-Bench, 57 Visium whole-transcriptome slides. **Stage 2:** HEST-1k training (~376 slides/fold), CRC test, mouse transfer. **Stage 3:** target noise ceiling, composition-aware SPARC-Map, niche and stage contrasts, and an external metastasis cohort (MD Anderson, GSE294385).

Key results (depth-adjusted Pearson r, mean over 16 CRC patient groups, H-optimus-1; full list in [`docs/CORE_FINDINGS.md`](docs/CORE_FINDINGS.md)):

| Finding | Number |
|---|---|
| Raw ssGSEA correlation is mostly non-specific: random size-matched gene sets reach 88% of curated raw r | 0.466 vs 0.408 |
| After depth adjustment, ULM targets are program-specific | 0.353 vs 0.095 random (16/16) |
| ULM targets beat SPARC-style ssGSEA targets | +0.109 (16/16) |
| Encoder is not the bottleneck: H-optimus-1 − H-optimus-0 | +0.008 (HEST-1k, n.s.); +0.027 (CRC-only, 11/16) |
| Headroom: achieved accuracy vs split-half reliability ceiling of measured ULM targets | 0.353 of 0.747 (47%) |
| Composition-aware two-stage model (composition + residual) vs direct prediction | 0.376 vs 0.348 (+0.028, 13/16) |
| Residual (beyond composition and depth) is predictable from H&E | 0.273 vs 0.092 random (16/16) |
| Within-tumour niche residuals are not program-specific on their own | 41 curated vs 71 random-set niche hits |
| MDACC liver metastasis stroma, micro vs macro: PD-1-high CD8 program shift is mostly residual (measured data) | residual +1.06 SD of +1.24; 10/10 patients; z 3.96 vs random max 1.32 |

Earlier H-optimus-0 results (18 groups: spatial-model variants, upgrades, nuclear-morphology adjustment, mouse) are in [`docs/RESULTS.md`](docs/RESULTS.md); they were not repeated with H-optimus-1.

Read [`docs/AUDIT.md`](docs/AUDIT.md) before reusing these numbers. It covers:
- a minor patient-level leakage in the training pool, which does not change the conclusions;
- the null sets being size-matched but not expression-matched;
- the residual being defined relative to a fixed scRNA reference;
- MDACC having no full-resolution H&E (measured decomposition only) and a small number of patients;
- mouse sections that cannot be split by animal.

## Layout

```
src/sparcpilot/        sparcmap_lib.py (SPARC-Map reproduction, training, metrics), encoders.py (H-optimus-1 / -0 loading, feature dirs),
                       spatial_models.py (MLP, CONCERT-bias, GP-kernel, Gaussian head), ssgsea.py (GPU ssGSEA)
scripts/
  setup_workspace.sh   builds the flat run directory the scripts expect and clones SPARC at the pinned commit
  00_data/             download_hest.py, download_models.py (h1 default), download_external.py (liver atlas, MDACC, COLON MAP),
                       build_programs.py, download_references.sh, build_splits.py, gene_panel.py
  01_features/         patch and 448 µm context embeddings (encoder from SPARC_ENCODER), extract_feats_dir.py (any patch dir)
  02_targets/          ssGSEA / ULM / MLM targets, random-set nulls, mouse orthologs, decoupling/ (scRNA reference, NNLS, residual targets)
  03_hestbench_pilot/  stage 1: targets/, spatial/ (B1 ablation, B2 heads), active_learning/, run_baseline.py
  04_hest1k/           stage 2 drivers (hest1k_run.py, hest1k_run2.py, crc_cv.py), h1/ (H-optimus-1 feature and training runs), hpc/ SLURM scripts
  05_mouse/            mouse transfer / retraining
  06_nuclei/           HoVer-Net spot features, CRC prediction dump
  07_analysis/         statistics and figures from results/raw/ and results/tables/
  08_headroom/         split-half (UMI and gene) noise ceiling of measured targets
  09_composition/      b_M reference baseline, composition-aware SPARC-Map, niche and COLON MAP stage contrasts, MDACC decomposition
metadata/              splits, slide lists, program and random gene sets, gene panel, ortholog table, MDACC sample → patient map
results/
  raw/                 per-slide × program metrics from every run (H-optimus-0: hest1k_wave*_long.parquet; H-optimus-1: hest1k_h1_long.parquet)
  tables/              summary and paired-test tables (top level = stage 2, H-optimus-0; h1/, headroom/, composition/, mdacc/ = stage 3)
  figures/             stage1_hestbench/, stage2_hest1k/, stage3_composition/, superseded/
docs/                  CORE_FINDINGS.md (current summary), METHODS.md, RESULTS.md (stage 1–2 record), AUDIT.md, stage1_hestbench/
environment/           pinned package versions (main analysis env, HoVer-Net env)
```

## Reproduce

Large inputs are not in git. They are downloaded by scripts or regenerated: HEST data is about 155 GB, model weights about 4.5 GB each, external cohorts about 11 GB, plus features and targets.

```bash
# 0. environments
conda create -n sparcpilot python=3.11 && pip install -r environment/requirements-main.txt
conda create -n nuclei python=3.11 && pip install -r environment/requirements-nuclei.txt      # only for 06_nuclei

# 1. flat working directory (scripts are symlinked, not copied; 09_composition scripts go to <workdir>/comp/)
bash scripts/setup_workspace.sh /path/to/workdir && cd /path/to/workdir
export HF_TOKEN=...            # needs access to MahmoodLab/hest and bioptimus/H-optimus-1 (gated); H-optimus-0 only for SPARC_ENCODER=h0
export SPARC_ENCODER=h1        # default; h0 reproduces the stage-2 H-optimus-0 runs
export SPARC_EXT=$PWD/ext      # external cohorts (default <workdir>/ext)

# 2. data, models, programs, references
python download_hest.py meta && python download_hest.py bench && python download_hest.py crc && python download_hest.py pool && python download_hest.py mouse
python download_models.py                     # h1 (default) or: python download_models.py h0
python build_programs.py                      # MSigDB 2026.2.Hs → sparc_programs_msigdb.json (36/43 programs)
bash download_references.sh                   # GSE132465 scRNA + MGI orthologs
python build_splits.py                        # reproduces the as-run split files byte-for-byte; add --strict for leakage-free pool

# 3. features and targets (GPU)
bash h1_features.sh                           # H-optimus-1: 58 CRC slides + HEST-1k pool, batched download/embed/delete
python extract_ctx.py && python extract_feats_mouse.py
python compute_targets_crc.py && python compute_targets_pool.py && python score_random_crc.py && python score_random_pool.py
python build_mgi_one2one.py && python compute_targets_mouse.py && python score_random_mouse.py

# 4. models — H-optimus-1 runs reported in docs/CORE_FINDINGS.md:
bash h1_train.sh crconly && bash h1_train.sh hest1k
#    stage-2 variants (H-optimus-0 era): python hest1k_run2.py --target ulm --seed 0 ; SLURM: scripts/04_hest1k/hpc/ (FEAT_TAG selects features)
python mouse_ft.py --seeds 0,1,2; for s in 0 1 2; do python crc_pred.py --seed $s; done
python hovernet_spots.py $(cat ffpe_crc_ids.txt)   # nuclei env

# 5. stage 3: headroom, composition, niches, external cohorts
python noise_ceiling.py && python analyze_noise_ceiling.py
python build_reference.py && python run_all.py sparc_programs_msigdb.json crc_folds.csv 16   # NNLS composition, 58 CRC slides (02_targets/decoupling)
python download_external.py                   # GSE115469, GSE294385 (+ authors' pathology layers; needs Rscript), COLON MAP obs
cd comp
python cache_gse132465.py && python decon_disjoint.py && python build_bM.py
for m in direct comp delta joint; do for s in 0 1 2; do python comp_model.py --mode $m --seed $s; done; done   # CRC-only, 6 folds
for m in comp delta; do python comp_model.py --mode $m --decon out_disjoint; done                               # gene-disjoint panel
for m in delta joint; do python comp_model.py --mode $m --target rand0ulm; done                                 # random-set null
python analyze_comp.py && python niche_decomp.py && python niche_null.py && python colonmap_stage.py
python build_ref17.py && python mdacc_prep.py
python build_bM_ref.py --ref ref17 --universe $SPARC_EXT/mdacc/universe.txt --tag _ref17_mdacc
python mdacc_decomp.py && python mdacc_null.py && python mdacc_patient.py
cd ..

# 6. statistics and figures (repo root; runs in seconds; no GPU or data needed)
python scripts/07_analysis/hest1k_stats.py && python scripts/07_analysis/mouse_stats.py && python scripts/07_analysis/nuclei_stats.py
python scripts/07_analysis/figures.py         # stage 2 → results/figures/regenerated/
python scripts/07_analysis/collect_h1_runs.py /path/to/workdir/results     # only after step 4
python scripts/07_analysis/compare_h0_h1.py && python scripts/07_analysis/figures_stage3.py
```

Step 6 alone reproduces the stage-2 numbers in `docs/RESULTS.md`, the encoder comparison and the stage-3 figures from the shipped `results/`. Each script in `scripts/09_composition/` lists its arguments in its docstring and in `scripts/09_composition/README.md`.

## Data and code sources

- **Data and models:**
  - HEST-1k / HEST-Bench (Jaume et al., MahmoodLab; CC BY-NC-SA 4.0), pinned revision in `download_hest.py`.
  - H-optimus-1 and H-optimus-0 (Bioptimus; gated on Hugging Face).
  - GSE132465 (Lee et al., Nat Genet 2020): CRC scRNA-seq reference.
  - GSE115469 (MacParland et al., Nat Commun 2018): human liver atlas, for the 17-type colon + liver reference.
  - COLON MAP Visium (Heiser et al., Cell 2023; OSF hftq2): pathologist annotations and stage, for the border-vs-core contrast.
  - GSE294385 and github.com/yliuup/CRC_micromets_ST (Liu et al., Cancer Cell 2026): CRC liver micro- and macrometastasis Visium with pathology layers; sample → patient map from the paper's Table S1.
  - MSigDB 2026.2.Hs; MGI orthology; HoVer-Net via TIAToolbox.
- **SPARC code:** SPARC (github.com/aziz-ayed/SPARC, CC BY-NC 4.0) is **not redistributed**. It is cloned at runtime by `setup_workspace.sh`, and `sparcmap_lib.py` imports its model definition.

## Status

This is exploratory pilot code, kept as run. It is not a packaged library. Any results should be cited as preliminary.

## License

Code, documentation, tables and figures in this repository: **CC BY-NC 4.0** (see `LICENSE`), © 2026 Zeyu Liu. This matches the upstream licences: SPARC is CC BY-NC 4.0, and HEST-1k is CC BY-NC-SA 4.0. The downloaded data and model weights remain under their own terms.
