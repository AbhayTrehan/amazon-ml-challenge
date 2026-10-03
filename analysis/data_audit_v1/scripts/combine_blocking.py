#!/usr/bin/env python3
"""Combine blocking channels on their common deterministic cohort; enumerate misses."""
import json,time
from collections import defaultdict
from pathlib import Path
import duckdb,pandas as pd,numpy as np
from common import fbeta,punct_norm,jaccard,dice_ngrams,tokens
from rapidfuzz.fuzz import ratio,token_set_ratio
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts';DB=OUT/'audit.duckdb'
start=time.time(); frames=[]
for channel in ['name_char','address_char']:
 for src in [2,3]:
  for country in ['India','US']:
   f=OUT/f'blocking_{channel}_s{src}_{country}.parquet'
   if f.exists():frames.append(pd.read_parquet(f))
retr=pd.concat(frames,ignore_index=True)
# Common cohort: exactly first 1,000 deterministic hash queries/country are present in every channel/source.
groups=[]
for country in ['India','US']:
 sets=[set(g.s1) for _,g in retr[retr.country==country].groupby(['channel','source'])]
 common=set.intersection(*sets); groups.extend(sorted(common))
qids=pd.DataFrame({'s1':groups});con=duckdb.connect(str(DB),read_only=True);con.register('qids',qids)
truth=con.execute("select q.s1,e.target_id from qids q left join edges e on q.s1=e.s1").fetchdf()
tmap={s:set(g.target_id.dropna()) for s,g in truth.groupby('s1')}
# Exact normalized candidates for cohort, country-preserving.
exact=con.execute("""select distinct s.entity_id s1,a.entity_id target_id from qids q join train_s1 s on q.s1=s.entity_id
 join (select * from train_s2 union all by name select * from train_s3)a on s.country=a.country and ((s.nname<>'' and s.nname=a.nname)or(s.naddr<>''and s.naddr=a.naddr))""").fetchdf()

def maps(frame): return {s:set(g.target_id) for s,g in frame.groupby('s1')}
emap=maps(exact); out=[]
for k in [1,2,3,5,10,20,50]:
 sub=retr[(retr.s1.isin(groups))&(retr['rank']<=k)]
 variants={'name':sub[sub.channel=='name_char'],'address':sub[sub.channel=='address_char'],'name_address':sub,'exact_address':pd.concat([sub[sub.channel=='address_char'][['s1','target_id']],exact]),'all_union':pd.concat([sub[['s1','target_id']],exact])}
 for label,fr in variants.items():
  pm=maps(fr.drop_duplicates());scores=[];te=hit=anyh=allh=0;counts=[]
  for s in groups:
   tr=tmap.get(s,set());pr=pm.get(s,set());te+=len(tr);hit+=len(tr&pr);anyh+=bool(tr)and bool(tr&pr);allh+=bool(tr)and tr<=pr;scores.append(fbeta(tr,tr&pr));counts.append(len(pr))
  out.append({'method':label,'k':k,'sample_queries':len(groups),'true_edges':te,'retrieved_edges':hit,'candidate_recall':hit/te,'candidate_pairs':sum(counts),'mean_candidates':np.mean(counts),'p50_candidates':np.quantile(counts,.5),'p95_candidates':np.quantile(counts,.95),'p99_candidates':np.quantile(counts,.99),'max_candidates':max(counts),'positive_any':anyh,'positive_all':allh,'oracle_macro_f05':np.mean(scores)})
pd.DataFrame(out).to_csv(OUT/'blocking_union_benchmarks.csv',index=False)

# Channel complementarity at K=50.
hitsets={}
for ch in ['name_char','address_char']:
 pm=maps(retr[(retr.s1.isin(groups))&(retr.channel==ch)&(retr['rank']<=50)])
 hitsets[ch]={(s,t) for s in groups for t in tmap.get(s,set()) if t in pm.get(s,set())}
ehit={(s,t) for s in groups for t in tmap.get(s,set()) if t in emap.get(s,set())}
alltruth={(s,t) for s in groups for t in tmap.get(s,set())}
comp=[]
for name,h in [('name_char',hitsets['name_char']),('address_char',hitsets['address_char']),('exact',ehit)]:
 others=set().union(*(x for n,x in [('name_char',hitsets['name_char']),('address_char',hitsets['address_char']),('exact',ehit)] if n!=name))
 comp.append({'channel':name,'true_links_recovered':len(h),'unique_true_links':len(h-others),'missed_true_links':len(alltruth-h),'candidate_pairs':len(retr[(retr.s1.isin(groups))&(retr.channel==name)&(retr['rank']<=50)]) if name!='exact' else len(exact)})
pd.DataFrame(comp).to_csv(OUT/'blocking_channel_complementarity.csv',index=False)

# Misses from strongest union.
best=maps(pd.concat([retr[(retr.s1.isin(groups))&(retr['rank']<=50)][['s1','target_id']],exact]).drop_duplicates())
miss=pd.DataFrame([(s,t) for s in groups for t in tmap.get(s,set()) if t not in best.get(s,set())],columns=['s1','target_id'])
con.register('miss0',miss)
missdetail=con.execute("""select m.*,case when starts_with(m.target_id,'S2-')then 2 else 3 end AS src,s.country,
 s.business_name s1_name,s.business_address s1_address,a.business_name target_name,a.business_address target_address
 from miss0 m join train_s1 s on m.s1=s.entity_id join (select * from train_s2 union all by name select * from train_s3)a on m.target_id=a.entity_id""").fetchdf()
for field,x,y in [('name','s1_name','target_name'),('address','s1_address','target_address')]:
 missdetail[field+'_ratio']=[ratio(punct_norm(a or''),punct_norm(b or''))/100 for a,b in zip(missdetail[x],missdetail[y])]
 missdetail[field+'_token_set']=[token_set_ratio(punct_norm(a or''),punct_norm(b or''))/100 for a,b in zip(missdetail[x],missdetail[y])]
 missdetail[field+'_jaccard']=[jaccard(a or'',b or'') for a,b in zip(missdetail[x],missdetail[y])]
 missdetail[field+'_char3_dice']=[dice_ngrams(a or'',b or'') for a,b in zip(missdetail[x],missdetail[y])]
def numjac(a,b):
 x={t for t in tokens(a or'')if t.isdigit()};y={t for t in tokens(b or'')if t.isdigit()};return len(x&y)/len(x|y)if x or y else 1.0
missdetail['numeric_jaccard']=[numjac(a,b)for a,b in zip(missdetail.s1_address,missdetail.target_address)]
missdetail.to_csv(OUT/'blocking_misses.tsv',sep='\t',index=False)
(OUT/'blocking_union_sampling.json').write_text(json.dumps({'method':'intersection of deterministic query cohorts: 1,000 S1 per country; full country-specific auxiliary indexes','seed':20260925,'queries':len(groups),'runtime_seconds':time.time()-start,'misses_best_union':len(miss)},indent=2))
print(pd.DataFrame(out).query("k==50").to_string(index=False));print('misses',len(miss))
