# sparc-pilot: null-calibrated evaluation of H&E → gene-program prediction

These are pilot experiments testing which upgrades to **SPARC** (H&E → spatial gene-program maps; Ayed et al., medRxiv 2026.05.04.26352410) are supported by data.

- **Data:** public HEST data only.
- **Stage 1:** HEST-Bench, 57 Visium whole-transcriptome slides.
- **Stage 2:** trained on HEST-1k, tested on 58 colorectal-cancer slides from 18 patient groups, plus 22 mouse sections.
- **Model and features:** a reproduction of the SPARC-Map spatial transformer on H-optimus-0 features.

Key results (depth-residualised Pearson r, CRC patient-group level; full tables in [`docs/RESULTS.md`](docs/RESULTS.md)):

| Finding | Number |
|---|---|
| Raw ssGSEA targets are mostly non-specific: random size-matched gene sets reach ~90% of real accuracy | 0.45 vs 0.40 raw |
| After depth residualisation, ULM targets are program-specific (36/36 programs > random) | 0.35 vs 0.09 (18/18) |
| ULM beats ssGSEA / MLM as training targets | +0.10 / +0.07 |
| Spatial context helps; richer spatial priors (CONCERT bias, GP kernels, 448 µm context) do not | −0.02 without context; others ±0.01, n.s. |
| Signal survives cell-type composition removal | 0.15 vs 0.03 random |
| Signal survives nuclear-morphology adjustment (HoVer-Net) | 91% retained; 0.32 vs 0.08 random |
| Human model transfers weakly to mouse; mouse retraining helps within dataset, not across datasets | 0.10 → 0.26 (slide CV); 0.10 (leave-dataset-out) |

Read [`docs/AUDIT.md`](docs/AUDIT.md) before reusing these numbers. It covers:
- a minor patient-level leakage in the training pool, which does not change the conclusions;
- the null sets being size-matched but not expression-matched;
- mouse sections that cannot be split by animal.

## Layout

```
src/sparcpilot/        sparcmap_lib.py (SPARC-Map reproduction, training, metrics), spatial_models.py (MLP, CONCERT-bias, GP-kernel, Gaussian head), ssgsea.py (GPU ssGSEA)
scripts/
  setup_workspace.sh   builds the flat run directory the scripts expect and clones SPARC at the pinned commit
  00_data/             download_hest.py, download_models.py, build_programs.py, download_references.sh, build_splits.py, gene_panel.py
  01_features/         H-optimus-0 patch and 448 µm context embeddings
  02_targets/          ssGSEA / ULM / MLM targets, random-set nulls, mouse orthologs, decoupling/ (scRNA reference, NNLS, residual targets)
  03_hestbench_pilot/  stage 1: targets/, spatial/ (B1 ablation, B2 heads), active_learning/, run_baseline.py
  04_hest1k/           stage 2 drivers (hest1k_run.py wave 1, hest1k_run2.py wave 2, crc_cv.py) and hpc/ SLURM scripts
  05_mouse/            mouse transfer / retraining
  06_nuclei/           HoVer-Net spot features, CRC prediction dump
  07_analysis/         statistics and figures from results/raw/
metadata/              splits, slide lists, program and random gene sets, gene panel, ortholog table (small, versioned)
results/
  raw/                 per-slide × program metrics from every run (parquet/CSV, ~8 MB) — input to 07_analysis
  tables/              summary and paired-test tables (stage1_hestbench/, stage2_inputs/, top level = stage 2)
  figures/             stage1_hestbench/, stage2_hest1k/ (final versions), superseded/
docs/                  METHODS.md, RESULTS.md, AUDIT.md, stage1_hestbench/ sub-reports
environment/           pinned package versions (main analysis env, HoVer-Net env)
```

## Reproduce

Large inputs are not in git. They are downloaded by scripts or regenerated: HEST data is about 155 GB, model weights about 4.3 GB, plus features and targets.

```bash
# 0. environments
conda create -n sparcpilot python=3.11 && pip install -r environment/requirements-main.txt
conda create -n nuclei python=3.11 && pip install -r environment/requirements-nuclei.txt      # only for 06_nuclei

# 1. flat working directory (scripts were run from one directory; they are symlinked, not copied)
bash scripts/setup_workspace.sh /path/to/workdir && cd /path/to/workdir
export HF_TOKEN=...            # needs access to MahmoodLab/hest and bioptimus/H-optimus-0 (gated)

# 2. data, models, programs, references
python download_hest.py meta && python download_hest.py bench && python download_hest.py crc && python download_hest.py pool && python download_hest.py mouse
python download_models.py
python build_programs.py                      # MSigDB 2026.2.Hs → sparc_programs_msigdb.json (36/43 programs)
bash download_references.sh                   # GSE132465 scRNA + MGI orthologs
python build_splits.py                        # reproduces the as-run split files byte-for-byte; add --strict for leakage-free pool

# 3. features and targets (GPU)
python extract_feats_crc.py && python extract_feats_pool.py && python extract_ctx.py && python extract_feats_mouse.py
python compute_targets_crc.py && python compute_targets_pool.py && python score_random_crc.py && python score_random_pool.py
python build_mgi_one2one.py && python compute_targets_mouse.py && python score_random_mouse.py

# 4. models — locally:  python hest1k_run2.py --target ulm --seed 0
#    on SLURM: see scripts/04_hest1k/hpc/ (set SPARC_WS, HPC_HOST, HPC_SCRATCH; account placeholder YOUR_ACCOUNT)
python mouse_ft.py --seeds 0,1,2; for s in 0 1 2; do python crc_pred.py --seed $s; done
python hovernet_spots.py $(cat ffpe_crc_ids.txt)   # nuclei env

# 5. statistics and figures from results/raw/ (runs in seconds; no GPU or data needed)
python scripts/07_analysis/hest1k_stats.py && python scripts/07_analysis/mouse_stats.py && python scripts/07_analysis/nuclei_stats.py
python scripts/07_analysis/figures.py         # → results/figures/regenerated/
```

Step 5 alone reproduces every number in `docs/RESULTS.md` from the shipped `results/raw/`.

## Data and code sources

- **Data and models:**
  - HEST-1k / HEST-Bench (Jaume et al., MahmoodLab; CC BY-NC-SA 4.0), pinned revision in `download_hest.py`.
  - H-optimus-0 (Bioptimus).
  - GSE132465 (Lee et al. 2020).
  - MSigDB 2026.2.Hs.
  - MGI orthology.
  - HoVer-Net via TIAToolbox.
- **SPARC code:** SPARC (github.com/aziz-ayed/SPARC, CC BY-NC 4.0) is **not redistributed**. It is cloned at runtime by `setup_workspace.sh`, and `sparcmap_lib.py` imports its model definition.

## Status

This is exploratory pilot code, kept as run. It is not a packaged library. Any results should be cited as preliminary.

## License

Code, documentation, tables and figures in this repository: **CC BY-NC 4.0** (see `LICENSE`), © 2026 Zeyu Liu. This matches the upstream licences: SPARC is CC BY-NC 4.0, and HEST-1k is CC BY-NC-SA 4.0. The downloaded data and model weights remain under their own terms.
