# Composition-aware decomposition (stage 3, H-optimus-1)

`setup_workspace.sh` links these scripts into `<workdir>/comp/`; they resolve the workspace as the parent of that folder and write to `comp/`, `comp/preds/` and `comp/results/`.
External cohorts are read from `$SPARC_EXT` (default `<workdir>/ext`; fill it with `python download_external.py`). Run order is in the top-level README, step 5.

| Script | Purpose | Main output |
|---|---|---|
| cache_gse132465.py | Sparse GSE132465 cell × gene matrix with 12-type labels; links `ref_profiles.parquet` and `out_parquet/` from the workspace | comp/gse132465_* |
| decon_disjoint.py | NNLS deconvolution with all program and random-set genes removed from the marker panel | comp/out_disjoint/ |
| build_bM.py | Pseudospot simulation from GSE132465 and the composition-and-depth baseline b_M (12 types) | comp/bM_{ulm,rand0ulm}.pt, bM_fit.csv |
| comp_model.py | `--mode direct / comp / delta / joint` SPARC-Map variants (CRC-only, 6 folds); `--decon`, `--target`, `--seed` | comp/preds/<mode>__<target>__<decon>__s<seed>/ |
| analyze_comp.py | Total, composition and residual accuracy; contrasts over 16 patient groups | results/comp_long.csv, comp_contrasts.csv |
| niche_decomp.py / niche_null.py | H&E niches; exact composition + depth + residual split; random-set and leave-one-out calibration | results/niche_*.csv |
| colonmap_stage.py | COLON MAP border-vs-core decomposition, stage III/IV vs I/II (OSF hftq2 annotations) | results/colonmap_*.csv |
| build_ref17.py | 17-type colon + liver reference (GSE132465 + GSE115469; counts recovered from the log-normalised liver matrix) | comp/ref17_* |
| mdacc_prep.py | GSE294385 sections: ULM scores, depth, NNLS composition (full and gene-disjoint panels) | $SPARC_EXT/mdacc/proc/, universe.txt |
| build_bM_ref.py | b_M for a chosen reference and gene universe (`--ref ref17 --universe … --tag _ref17_mdacc`) | comp/bM_*_ref17_mdacc.pt |
| mdacc_decomp.py | Composition / depth / residual split of micro- vs macrometastasis, primary vs liver and front vs superficial contrasts (section level) | results/mdacc_contrasts_*.csv |
| mdacc_null.py | Random-set-calibrated z-scores per section; leave-one-out calibration of the random null | results/mdacc_null_calibrated.csv |
| mdacc_patient.py | Patient-level version (Table S1 map, `metadata/mdacc_patient_map.csv`) | results/mdacc_patient_level.csv |

Shipped copies of the outputs: `results/tables/composition/`, `results/tables/mdacc/`. Findings: `docs/CORE_FINDINGS.md`, sections 8–10.
`scripts/04_hest1k/hest1k_run_x.py` extends the HEST-1k driver with target smoothing and per-spot prediction dumps (not used for reported results).
