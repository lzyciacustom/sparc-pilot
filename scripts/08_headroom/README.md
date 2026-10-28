# Target noise ceiling (stage 3)

Linked flat into the workspace (other scripts import `noise_ceiling.ulm` / `lognorm`).

- `noise_ceiling.py`: per CRC test section, splits measured counts into two binomial halves (UMI split, technical noise) and splits each program's genes into halves (gene split); scores both halves (ULM, ssGSEA, random sets); Spearman–Brown corrects the split-half correlation after depth adjustment. Output: `results/noise_ceiling_long.csv`.
- `analyze_noise_ceiling.py`: ceiling (√ reliability) vs achieved H&E accuracy by program and patient group. Outputs: `results/noise_ceiling_by_{program,group}.csv`.

Shipped copies: `results/tables/headroom/`. Findings: `docs/CORE_FINDINGS.md`, section 7.
