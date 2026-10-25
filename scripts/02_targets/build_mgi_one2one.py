"""Strict one-to-one human-mouse orthologs from MGI HOM_MouseHumanSequence.rpt -> mgi_one2one_human_mouse.csv.
Classes with exactly one human and one mouse row are paired; pairs are kept only if each symbol belongs to a single
homology class and appears in a single pair (pilot: 17,090 pairs)."""
import sys, pandas as pd
src = sys.argv[1] if len(sys.argv) > 1 else 'data/ref/mgi/HOM_MouseHumanSequence.rpt'
h = pd.read_csv(src, sep='\t', dtype=str); K = 'DB Class Key'
hs = h[h['NCBI Taxon ID'] == '9606']; mm = h[h['NCBI Taxon ID'] == '10090']
nh = hs.groupby(K).size(); nm = mm.groupby(K).size()
one = set(nh[nh == 1].index) & set(nm[nm == 1].index)
m = hs[hs[K].isin(one)][[K, 'Symbol']].merge(mm[mm[K].isin(one)][[K, 'Symbol']], on=K, suffixes=('_hs', '_mm'))
hc = hs.groupby('Symbol')[K].nunique(); mc = mm.groupby('Symbol')[K].nunique()
m1 = m[m.Symbol_hs.map(hc).eq(1) & m.Symbol_mm.map(mc).eq(1)]
m1 = m1[~m1.Symbol_hs.duplicated(keep=False) & ~m1.Symbol_mm.duplicated(keep=False)]
m1[['Symbol_hs', 'Symbol_mm', K]].rename(columns={'Symbol_hs': 'human_symbol', 'Symbol_mm': 'mouse_symbol', K: 'mgi_db_class_key'}).to_csv('mgi_one2one_human_mouse.csv', index=False)
print(len(m), 'candidate pairs ->', len(m1), 'strict 1:1')
