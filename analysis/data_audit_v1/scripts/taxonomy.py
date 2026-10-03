#!/usr/bin/env python3
"""Heuristic, explicitly sampled corruption/error taxonomy from local text only."""
from pathlib import Path
import pandas as pd
from common import punct_norm,suffix_norm,tokens
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';d=pd.read_parquet(OUT/'similarity_pair_sample.parquet');p=d[d.family=='positive'].copy()
def category(r):
 sn,an,sa,aa=[x or''for x in[r.s_name,r.a_name,r.s_addr,r.a_addr]]
 if sn==an and sa==aa:return'exact_both_raw'
 if (not sa or not aa)and not(r.addr_exact):return'address_missing_one_side'
 if (not sn or not an)and not(r.name_exact):return'name_missing_one_side'
 if r.name_exact and r.addr_exact:return'normalized_both_exact'
 if suffix_norm(sn)==suffix_norm(an)and punct_norm(sn)!=punct_norm(an):return'legal_suffix_only_name'
 if sorted(tokens(sn))==sorted(tokens(an))and punct_norm(sn)!=punct_norm(an):return'name_word_reordering'
 if r.name_exact:return'normalized_name_exact_only'
 if r.addr_exact:return'normalized_address_exact_only'
 if r.name_char_tfidf_cosine>=.75 and r.addr_char_tfidf_cosine>=.75:return'both_lexically_strong'
 if r.name_char_tfidf_cosine<.3 and r.addr_char_tfidf_cosine>=.7:return'name_heavy_corruption_address_recovers'
 if r.addr_char_tfidf_cosine<.3 and r.name_char_tfidf_cosine>=.7:return'address_heavy_corruption_name_recovers'
 if max(r.name_char_tfidf_cosine,r.addr_char_tfidf_cosine)<.3:return'both_lexically_weak'
 return'mixed_partial_corruption'
p['category']=[category(r)for r in p.itertuples()]
summ=p.groupby(['category','target_source','country']).size().rename('n').reset_index();summ['within_stratum_rate']=summ.n/20000;summ.to_csv(OUT/'positive_difficulty_taxonomy.csv',index=False)
p.groupby('category',group_keys=False).head(8)[['category','s1','target_id','target_source','country','s_name','s_addr','a_name','a_addr','name_char_tfidf_cosine','addr_char_tfidf_cosine','numeric_conflict']].to_csv(OUT/'taxonomy_examples.tsv',sep='\t',index=False)
print(p.category.value_counts().to_string())
