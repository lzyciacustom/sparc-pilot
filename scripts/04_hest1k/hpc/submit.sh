#!/bin/bash
set -euo pipefail
WS="${SPARC_WS:?set SPARC_WS to your local checkout/workspace}"
H="${HPC_HOST:?set HPC_HOST (ssh alias of the cluster login node)}"
R=${HPC_SCRATCH:?set HPC_SCRATCH}/sparc_pilot
remote() { ssh "$H" "bash -lc '$1'"; }
case "${1:-}" in
  check)  ssh -O check "$H" ;;
  upload)
    remote "mkdir -p $R/logs $R/ws/trackB $R/ws/sparc_repo/preprocessing $R/ws/hprc $R/ws/results $R/ws/features/hoptimus0_crc $R/ws/targets_crc"
    cd "$WS"
    rsync -a hest1k_run.py sparcmap_lib.py ens2sym.py crc_folds.csv hest1k_train_pool.csv hest1k_crc_candidates.csv hest_bench_inventory.csv sparc_programs_msigdb.json "$H:$R/ws/"
    rsync -a trackB/spatial_models.py "$H:$R/ws/trackB/"
    rsync -a sparc_repo/preprocessing/__init__.py sparc_repo/preprocessing/map_models.py "$H:$R/ws/sparc_repo/preprocessing/"
    rsync -a hprc/ "$H:$R/ws/hprc/"
    rsync -a --info=progress2 features/hoptimus0_crc/ "$H:$R/ws/features/hoptimus0_crc/"
    rsync -a --info=progress2 --include='*.parquet' --exclude='*' targets_crc/ "$H:$R/ws/targets_crc/"
    remote "du -sh $R/ws; ls $R/ws/features/hoptimus0_crc | wc -l; ls $R/ws/targets_crc | wc -l" ;;
  sync)   cd "$WS"; rsync -a hprc/ "$H:$R/ws/hprc/"; rsync -a hest1k_run.py sparcmap_lib.py "$H:$R/ws/"; echo synced ;;
  rerun)
    IDX="${2:?give comma-separated jobs.txt line numbers}"
    for i in ${IDX//,/ }; do read T M Hd Sd X < <(sed -n "${i}p" "$WS/hprc/jobs.txt"); TAG="${T}__${M}__${Hd}__s${Sd}${X:+__crconly}"; remote "rm -f $R/logs/done_${TAG}"; echo "cleared $TAG"; done
    remote "cd $R && sbatch --array=$IDX ws/hprc/run.slurm" ;;
  setup)  remote "cd $R && sbatch ws/hprc/setup.slurm" ;;
  smoke)  remote "cd $R && sbatch ws/hprc/smoke.slurm" ;;
  submit) N=$(wc -l < "$WS/hprc/jobs.txt"); remote "cd $R && sbatch --array=1-$N%8 ws/hprc/run.slurm" ;;
  status) remote "squeue -u \$USER; echo results: \$(ls $R/ws/results 2>/dev/null | wc -l); for f in \$(ls -t $R/logs/*.out 2>/dev/null | head -3); do echo == \$f; tail -n 4 \$f; done" ;;
  fetch)  mkdir -p "$WS/hpc_results/logs"; rsync -a "$H:$R/ws/results/" "$WS/hpc_results/"; rsync -a "$H:$R/logs/" "$WS/hpc_results/logs/"; ls "$WS/hpc_results" | wc -l ;;
  upload2)
    cd "$WS"; remote "mkdir -p $R/ws/genes_crc $R/ws/results2"
    rsync -a hest1k_run2.py sparcmap_lib.py gene_panel.json A2_splithalf_per_program.csv mouse_test_slides_meta.csv "$H:$R/ws/"
    rsync -a hprc/ "$H:$R/ws/hprc/"
    rsync -a --info=progress2 genes_crc/ "$H:$R/ws/genes_crc/"
    remote "ls $R/ws/genes_crc | wc -l; du -sh $R/ws/genes_crc" ;;
  upload2b)
    cd "$WS"; remote "mkdir -p $R/ws/features/hoptimus0_ctx $R/ws/features/hoptimus0_mouse $R/ws/decoup $R/ws/targets_mouse"
    rsync -a hest1k_run2.py sparcmap_lib.py mouse_test_slides_meta.csv "$H:$R/ws/"; rsync -a hprc/ "$H:$R/ws/hprc/"
    rsync -a --info=progress2 features/hoptimus0_ctx/ "$H:$R/ws/features/hoptimus0_ctx/"
    rsync -a --info=progress2 features/hoptimus0_mouse/ "$H:$R/ws/features/hoptimus0_mouse/"
    rsync -a decoup/ "$H:$R/ws/decoup/"; rsync -a targets_mouse/ "$H:$R/ws/targets_mouse/"
    remote "for d in features/hoptimus0_ctx features/hoptimus0_mouse decoup targets_mouse; do echo \$d \$(ls $R/ws/\$d | wc -l); done" ;;
  submit2)
    JF="${2:-jobs2.txt}"; N=$(wc -l < "$WS/hprc/$JF"); ARR="${3:-1-$N}"
    rsync -a "$WS/hprc/" "$H:$R/ws/hprc/"; rsync -a "$WS/hest1k_run2.py" "$WS/sparcmap_lib.py" "$H:$R/ws/"
    remote "cd $R && sbatch --array=$ARR%12 ws/hprc/run2.slurm $JF" ;;
  status2)   remote "squeue -u \$USER; echo results2: \$(ls $R/ws/results2 2>/dev/null | grep -v tmp | wc -l); grep -l Traceback $R/logs/sparc_r2_*.out 2>/dev/null | head; for f in \$(ls -t $R/logs/sparc_r2_*.out 2>/dev/null | head -3); do echo == \$f; tail -n 3 \$f; done" ;;
  fetch2)    mkdir -p "$WS/hpc_results2/logs"; rsync -a "$H:$R/ws/results2/" "$WS/hpc_results2/"; rsync -a --include='sparc_r2_*' --exclude='*' "$H:$R/logs/" "$WS/hpc_results2/logs/"; ls "$WS/hpc_results2" | wc -l ;;
  *) echo "usage: bash submit.sh check|upload|sync|rerun <idx>|setup|smoke|submit|status|fetch|upload2|upload2b|submit2 <jobsfile> [array]|status2|fetch2"; exit 1 ;;
esac
