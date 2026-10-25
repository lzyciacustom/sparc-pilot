"""Build sparc_programs_msigdb.json: the SPARC program list (sparc_repo/preprocessing/sparc_map.py, 43 names) matched to MSigDB.
Pilot used MSigDB 2026.2.Hs (Hallmark, C2:CP:Reactome, C7:ImmuneSigDB): 36 found; 7 literature signatures absent
(ANTIGEN_PRESENTATION_THOMPSON_2020_APM8, B_CELL_CORE_BUDCZIES_2021, CIN70_CHROMOSOMAL_INSTABILITY, TGFb_STROMAL_EXCLUSION_MARIATHASAN_2018,
TLS_CABRITA_9, T_CELL_INFLAMED_GEP_18_AYERS_2017, REVSC). The released json is in metadata/ (use it for exact reproduction)."""
import gseapy, json, re, sys
ver = sys.argv[1] if len(sys.argv) > 1 else '2026.2.Hs'
m = gseapy.Msigdb(); gs = {}
for cat in ['h.all', 'c2.cp.reactome', 'c7.immunesigdb']: gs.update(m.get_gmt(category=cat, dbver=ver))
src = open('sparc_repo/preprocessing/sparc_map.py').read()
PROGRAMS = re.findall(r'^\s+"([A-Za-z0-9_]+)",\s*$', src.split('PROGRAMS = [')[1].split(']')[0], flags=re.M)
found = {p: gs[p] for p in PROGRAMS if p in gs}
print(len(PROGRAMS), 'listed;', len(found), 'found; missing:', [p for p in PROGRAMS if p not in gs])
json.dump({'msigdb_version': ver, 'sets': found}, open('sparc_programs_msigdb.json', 'w'))
