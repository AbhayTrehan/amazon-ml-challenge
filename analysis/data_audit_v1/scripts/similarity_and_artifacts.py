#!/usr/bin/env python3
"""Deterministic sampled rich similarity diagnostics plus exact all-edge ID tests."""
import json,time,hashlib
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from rapidfuzz.fuzz import ratio,partial_ratio,token_sort_ratio,token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from common import punct_norm,tokens,levenshtein_ratio,jaccard,dice_ngrams

ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'analysis/data_audit_v1/artifacts'; DB=OUT/'audit.duckdb'; SEED=20260925
con=duckdb.connect(str(DB),read_only=False); con.execute("PRAGMA threads=6");con.execute("PRAGMA memory_limit='3GB'")
start=time.time()

union_aux="(select 2 src,* from train_s2 union all by name select 3 src,* from train_s3)"
base="""select e.s1,e.target_id,e.target_source,s.country,s.business_name s_name,s.business_address s_addr,
 a.business_name a_name,a.business_address a_addr,s.row_num s_row,a.row_num a_row
 from edges e join train_s1 s on e.s1=s.entity_id join %s a on e.target_id=a.entity_id and e.target_source=a.src"""%union_aux
pos=con.execute("select * exclude(rn) from (select *,row_number() over(partition by target_source,country order by hash(s1,target_id,?))rn from ("+base+")) where rn<=20000",[SEED]).fetchdf()
pos['family']='positive'

# Same-country random negatives from deterministic 30k-record auxiliary reservoirs.
aux_sample=con.execute("select * exclude(rn) from (select src,entity_id,business_name,business_address,country,row_num,row_number()over(partition by src,country order by hash(entity_id,?))rn from "+union_aux+")where rn<=30000",[SEED]).fetchdf()
rrows=[]
aux_groups={(int(k[0]),k[1]):g.reset_index(drop=True) for k,g in aux_sample.groupby(['src','country'])}
for p in pos[['s1','target_source','country','s_name','s_addr','s_row']].drop_duplicates().itertuples(index=False):
 a=aux_groups[(int(p.target_source),p.country)]
 ix=int.from_bytes(hashlib.blake2b(f'{p.s1}|{SEED}'.encode(),digest_size=8).digest(),'big')%len(a); q=a.iloc[ix]
 rrows.append((p.s1,q.entity_id,p.target_source,p.country,p.s_name,p.s_addr,q.business_name,q.business_address,p.s_row,q.row_num))
rand=pd.DataFrame(rrows,columns=['s1','target_id','target_source','country','s_name','s_addr','a_name','a_addr','s_row','a_row'])
con.register('rand0',rand); rand=con.execute("select r.* from rand0 r left join edges e on r.s1=e.s1 and r.target_id=e.target_id where e.target_id is null").fetchdf();rand['family']='random_same_country'

# Exact-normalized-name collision negatives; cap deterministically to 20k/stratum.
coll=con.execute("""WITH ss AS (SELECT * exclude(sr) FROM (SELECT *,row_number()over(partition by country order by hash(entity_id,20260925))sr FROM train_s1) WHERE sr<=50000)
 select * exclude(rn) from (select s.entity_id s1,a.entity_id target_id,a.src target_source,s.country,s.business_name s_name,s.business_address s_addr,
 a.business_name a_name,a.business_address a_addr,s.row_num s_row,a.row_num a_row,
 row_number()over(partition by a.src,s.country order by hash(s.entity_id,a.entity_id,20260925))rn
 from ss s join %s a on s.country=a.country and s.nname<>'' and s.nname=a.nname
 left join edges e on e.s1=s.entity_id and e.target_id=a.entity_id where e.target_id is null)where rn<=20000"""%union_aux).fetchdf();coll['family']='normalization_collision'
df=pd.concat([pos,rand,coll],ignore_index=True)

def feats(r):
 n1,n2=punct_norm(r.s_name or ''),punct_norm(r.a_name or ''); a1,a2=punct_norm(r.s_addr or ''),punct_norm(r.a_addr or '')
 nums1={x for x in tokens(a1) if x.isdigit()};nums2={x for x in tokens(a2) if x.isdigit()}
 return [n1==n2 and n1!='',a1==a2 and a1!='',ratio(n1,n2)/100,partial_ratio(n1,n2)/100,token_sort_ratio(n1,n2)/100,token_set_ratio(n1,n2)/100,jaccard(n1,n2),dice_ngrams(n1,n2),
 ratio(a1,a2)/100,partial_ratio(a1,a2)/100,token_sort_ratio(a1,a2)/100,token_set_ratio(a1,a2)/100,jaccard(a1,a2),dice_ngrams(a1,a2),
 len(nums1&nums2)/len(nums1|nums2) if nums1 or nums2 else 1.0, bool(nums1&nums2),bool(nums1 and nums2 and not nums1&nums2)]
cols=['name_exact','addr_exact','name_ratio','name_partial','name_token_sort','name_token_set','name_jaccard','name_char3_dice','addr_ratio','addr_partial','addr_token_sort','addr_token_set','addr_jaccard','addr_char3_dice','numeric_jaccard','numeric_any_overlap','numeric_conflict']
df[cols]=pd.DataFrame([feats(r) for r in df.itertuples()],columns=cols)

# Corpus-IDF pairwise cosines for word and char views on this deterministic diagnostic sample.
for prefix,xcol,ycol in [('name','s_name','a_name'),('addr','s_addr','a_addr')]:
 texts=pd.concat([df[xcol],df[ycol]]).fillna('').map(punct_norm)
 for label,kwargs in [('word_tfidf',dict(analyzer='word',ngram_range=(1,2),min_df=2,max_features=250000)),('char_tfidf',dict(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=250000))]:
  v=TfidfVectorizer(dtype=np.float32,**kwargs); X=v.fit_transform(texts); n=len(df)
  df[f'{prefix}_{label}_cosine']=np.asarray(X[:n].multiply(X[n:]).sum(axis=1)).ravel()

feature_cols=cols+['name_word_tfidf_cosine','name_char_tfidf_cosine','addr_word_tfidf_cosine','addr_char_tfidf_cosine']
df.to_parquet(OUT/'similarity_pair_sample.parquet',index=False)
qs=[.01,.05,.1,.25,.5,.75,.9,.95,.99]
records=[]
for keys,g in df.groupby(['family','target_source','country'],dropna=False):
 for c in feature_cols:
  vals=g[c].astype(float)
  rec={'family':keys[0],'target_source':keys[1],'country':keys[2],'feature':c,'n':len(g),'mean':vals.mean()}
  rec.update({f'p{int(q*100)}':vals.quantile(q) for q in qs});records.append(rec)
pd.DataFrame(records).to_csv(OUT/'similarity_distributions.csv',index=False)

disc=[]
for neg in ['random_same_country','normalization_collision']:
 z=df[df.family.isin(['positive',neg])]
 y=(z.family=='positive').astype(int)
 for c in feature_cols:
  try: auc=roc_auc_score(y,z[c].astype(float))
  except: auc=None
  disc.append({'negative_family':neg,'feature':c,'roc_auc':auc,'positive_mean':df[df.family=='positive'][c].mean(),'negative_mean':df[df.family==neg][c].mean()})
pd.DataFrame(disc).to_csv(OUT/'feature_discriminability.csv',index=False)

# Hard examples: lowest best-channel score positives, highest combined collision negatives.
df['best_lex']=df[['name_char_tfidf_cosine','addr_char_tfidf_cosine','name_token_set','addr_token_set']].max(axis=1)
df['combined']=.55*df.name_char_tfidf_cosine+.45*df.addr_char_tfidf_cosine
keep=['s1','target_id','target_source','country','s_name','s_addr','a_name','a_addr']+feature_cols
df[df.family=='positive'].nsmallest(250,'best_lex')[keep].to_csv(OUT/'hard_positive_examples.tsv',sep='\t',index=False)
df[df.family!='positive'].nlargest(250,'combined')[['family']+keep].to_csv(OUT/'hard_negative_examples.tsv',sep='\t',index=False)

# Exact all-edge ID and row-order signals versus deterministic same-country random negative.
idstats=con.execute("""WITH p AS (SELECT e.target_source,s.country,
 abs(try_cast(split_part(s.entity_id,'-',2)as bigint)-try_cast(split_part(a.entity_id,'-',2)as bigint)) iddiff,
 abs(s.row_num::double/(select count(*)from train_s1)-a.row_num::double/case when e.target_source=2 then (select count(*)from train_s2)else(select count(*)from train_s3)end) pctdiff,
 split_part(s.entity_id,'-',2)=split_part(a.entity_id,'-',2) same_suffix
 FROM edges e join train_s1 s on e.s1=s.entity_id join %s a on e.target_id=a.entity_id and e.target_source=a.src)
 SELECT target_source,country,count(*)n,avg(iddiff)mean_iddiff,quantile_disc(iddiff,.01)p1,quantile_disc(iddiff,.5)p50,quantile_disc(iddiff,.99)p99,
 avg(pctdiff)mean_pctdiff,quantile_disc(pctdiff,.01)pct_p1,quantile_disc(pctdiff,.5)pct_p50,quantile_disc(pctdiff,.99)pct_p99,count(*)filter(where same_suffix)same_suffix
 FROM p GROUP BY all ORDER BY 1,2"""%union_aux).fetchdf()
idstats.to_csv(OUT/'id_order_artifacts.csv',index=False)

meta={'seed':SEED,'positive_sample_method':'up to 20,000 per source x country by smallest deterministic hash','positive_n':int((df.family=='positive').sum()),'random_n':int((df.family=='random_same_country').sum()),'collision_n':int((df.family=='normalization_collision').sum()),'runtime_seconds':time.time()-start}
(OUT/'similarity_sampling.json').write_text(json.dumps(meta,indent=2))
print(meta)
