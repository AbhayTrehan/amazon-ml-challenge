#!/usr/bin/env python3
"""Post-hoc D002 diagnostics from clean predictions and existing P002 caches."""
from __future__ import annotations
import json,math,re,sys,time,unicodedata
from pathlib import Path
import duckdb,numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"analysis/experiments_v1/d002_clean_public_gap"
P002=ROOT/"analysis/production/p002_final"
E001=ROOT/"analysis/experiments_v1/e001_classical_baseline"
E005=ROOT/"analysis/experiments_v1/e005_xlmr_crossencoder"
R007=ROOT/"analysis/experiments_v1/r007_fast_blocker_e005b"
DB=ROOT/"analysis/data_audit_v1/artifacts/audit.duckdb"
ASSIGN=E001/"cache/auxiliary_fold_assignment.parquet"
TEST=ROOT/"student_resource/dataset/test"
Q=[.10,.25,.50,.75,.90,.95,.99]
BINS=np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,1.0000001])
LABELS=["[0,.1)","[.1,.2)","[.2,.3)","[.3,.4)","[.4,.5)","[.5,.6)","[.6,.7)","[.7,.8)","[.8,.9)","[.9,.95)","[.95,.98)","[.98,1]"]

def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(v,indent=2,default=float)+"\n");q.replace(p)
def pct_rows(pop,kind,a):
 z=np.quantile(np.asarray(a,float),Q) if len(a) else np.full(len(Q),np.nan)
 return [{"population":pop,"statistic":kind,**{f"p{int(q*100)}":v for q,v in zip(Q,z)}}]
def f05_ideal_precision(pred,true):
 return 1.25*pred/(pred+.25*true) if true>0 else 1.

def score_and_calibration():
 start=time.time();clean=pd.read_parquet(OUT/"output/clean_predictions.parquet")
 rows=[];reliability=[];expected=[];pair_rates=[]
 cal=pd.read_parquet(E005/"outputs/calibration_predictions.parquet",columns=["label","lightgbm_score","neural_probability"])
 cal_score=.6*cal.neural_probability.to_numpy(float)+.4*cal.lightgbm_score.to_numpy(float)
 iso=IsotonicRegression(out_of_bounds="clip").fit(cal_score,cal.label.to_numpy())
 for country in ["US","India"]:
  d=clean[clean.country.eq(country)];pop=f"clean {country}"
  rows+=pct_rows(pop,"lightgbm_shortlist",d.lightgbm_score);rows+=pct_rows(pop,"xlmr_shortlist",d.neural_probability);rows+=pct_rows(pop,"fused_shortlist",d.fused_score)
  w=d[d.owner_rank.eq(1)];rows+=pct_rows(pop,"owner_winning_score",w.fused_score);rows+=pct_rows(pop,"owner_margin",w.owner_margin)
  n=d.source1_entity_id.nunique();above=int(d.fused_score.ge(.51).sum());winabove=int((d.owner_rank.eq(1)&d.fused_score.ge(.51)).sum());marginrej=int((d.owner_rank.eq(1)&d.fused_score.ge(.51)&d.owner_margin.lt(.48)).sum());accepted=int(d.accepted.sum())
  pair_rates.append({"population":pop,"s1":n,"above_threshold_per_s1":above/n,"rejected_competition_per_s1":(above-winabove)/n,"rejected_margin_only_per_s1":marginrej/n,"accepted_per_s1":accepted/n})
 # Clean reliability, common across countries.
 idx=np.clip(np.digitize(clean.fused_score.to_numpy(),BINS,right=False)-1,0,len(LABELS)-1)
 for i,label in enumerate(LABELS):
  take=idx==i;reliability.append({"population":"clean fold0","bin":label,"pair_count":int(take.sum()),"empirical_true_positive_rate":float(clean.label.to_numpy()[take].mean()) if take.any() else np.nan})

 # Full test pair arrays are processed one country at a time from existing shards.
 for country in ["US","India","France"]:
  lg=[];nn=[];fu=[];bin_counts=np.zeros(len(LABELS),np.int64);exp_sum=0.;above=0;nrows=0
  for sp in sorted((P002/f"cache/shortlist/{country}").glob("*.parquet")):
   npth=P002/f"cache/neural/{country}/{sp.name}";s=pd.read_parquet(sp);n=pd.read_parquet(npth)
   if not s[["source1_entity_id","candidate_entity_id","candidate_source"]].equals(n[["source1_entity_id","candidate_entity_id","candidate_source"]]):raise RuntimeError(f"test key mismatch {sp}")
   a=s.lightgbm_score.to_numpy(np.float32);b=n.neural_probability.to_numpy(np.float32);f=(.6*b+.4*a).astype(np.float32)
   lg.append(a);nn.append(b);fu.append(f);j=np.clip(np.digitize(f,BINS)-1,0,len(LABELS)-1);bin_counts+=np.bincount(j,minlength=len(LABELS));exp_sum+=float(iso.predict(f).sum());above+=int((f>=.51).sum());nrows+=len(f)
  lg=np.concatenate(lg);nn=np.concatenate(nn);fu=np.concatenate(fu);pop=f"test {country}";rows+=pct_rows(pop,"lightgbm_shortlist",lg);rows+=pct_rows(pop,"xlmr_shortlist",nn);rows+=pct_rows(pop,"fused_shortlist",fu)
  for i,label in enumerate(LABELS):reliability.append({"population":pop,"bin":label,"pair_count":int(bin_counts[i]),"empirical_true_positive_rate":np.nan})
  ns1={"US":663106,"India":809986,"France":259452}[country];accepted={"US":2144142,"India":2496737,"France":686518}[country]
  expected.append({"country":country,"s1":ns1,"accepted_edges":accepted,"accepted_edges_per_s1":accepted/ns1,"calibrated_expected_edges":exp_sum,"calibrated_expected_edges_per_s1":exp_sum/ns1,"gap_expected_minus_accepted_per_s1":(exp_sum-accepted)/ns1,"confidence":"OOD / low" if country=="France" else "in-domain diagnostic"})
  pair_rates.append({"population":pop,"s1":ns1,"above_threshold_per_s1":above/ns1})
  del lg,nn,fu

 # Test owner diagnostics, once, using the same exact definitions as validation.
 con=duckdb.connect();con.execute("pragma threads=20");con.execute("pragma memory_limit='100GB'");tmp=OUT/"cache/ducktmp_analysis";tmp.mkdir(parents=True,exist_ok=True);con.execute(f"pragma temp_directory='{tmp}'")
 sg=str(P002/"cache/shortlist/*/*.parquet");ng=str(P002/"cache/neural/*/*.parquet");owner_path=OUT/"output/test_owner_diagnostics.parquet"
 if not owner_path.exists():
  con.execute(f"""copy(with scored as(select s.source1_entity_id,s.candidate_entity_id,s.country,(.6*n.neural_probability+.4*s.lightgbm_score)::float fused_score from read_parquet('{sg}',union_by_name=true)s join read_parquet('{ng}',union_by_name=true)n using(source1_entity_id,candidate_entity_id,candidate_source)), ranked as(select *,row_number()over(partition by candidate_entity_id order by fused_score desc,source1_entity_id)owner_rank,lead(fused_score)over(partition by candidate_entity_id order by fused_score desc,source1_entity_id)second_score from scored)select source1_entity_id,candidate_entity_id,country,fused_score,(fused_score-coalesce(second_score,0))::float owner_margin from ranked where owner_rank=1)to '{owner_path}'(format parquet,compression zstd)""")
 owners=pd.read_parquet(owner_path)
 for country in ["US","India","France"]:
  x=owners[owners.country.eq(country)];pop=f"test {country}";rows+=pct_rows(pop,"owner_winning_score",x.fused_score);rows+=pct_rows(pop,"owner_margin",x.owner_margin)
  ns1={"US":663106,"India":809986,"France":259452}[country];above=next(r["above_threshold_per_s1"] for r in pair_rates if r["population"]==pop)*ns1;winabove=int(x.fused_score.ge(.51).sum());marginrej=int((x.fused_score.ge(.51)&x.owner_margin.lt(.48)).sum());accepted=int((x.fused_score.ge(.51)&x.owner_margin.ge(.48)).sum())
  r=next(r for r in pair_rates if r["population"]==pop);r.update({"rejected_competition_per_s1":(above-winabove)/ns1,"rejected_margin_only_per_s1":marginrej/ns1,"accepted_per_s1":accepted/ns1})
 con.close();pd.DataFrame(rows).to_csv(OUT/"output/score_percentiles.csv",index=False);pd.DataFrame(pair_rates).to_csv(OUT/"output/decoder_rates.csv",index=False);pd.DataFrame(reliability).to_csv(OUT/"output/calibration_by_score.csv",index=False);pd.DataFrame(expected).to_csv(OUT/"output/expected_edge_count.csv",index=False)
 atomic_json(OUT/"runtime/score_calibration.json",{"seconds":time.time()-start,"status":"complete","isotonic_fit":"cached fold4 E005 predictions"})

def structural():
 start=time.time();clean=pd.read_parquet(OUT/"output/per_s1_scores.parquet");res=json.loads((OUT/"output/clean_results.json").read_text());con=duckdb.connect(str(DB),read_only=True)
 s1_test=dict(con.execute("select country,count(*) from test_s1 group by 1").fetchall());aux_test={(c,int(s)):n for c,s,n in con.execute("select country,2,count(*)from test_s2 group by 1 union all select country,3,count(*)from test_s3 group by 1").fetchall()}
 aux_clean={(c,int(s)):n for c,s,n in con.execute(f"""select a.country,2,count(*)from train_s2 a join read_parquet('{ASSIGN}')f on f.source=2 and f.entity_id=a.entity_id where f.fold=0 group by 1 union all select a.country,3,count(*)from train_s3 a join read_parquet('{ASSIGN}')f on f.source=3 and f.entity_id=a.entity_id where f.fold=0 group by 1""").fetchall()};con.close()
 rows=[]
 for country in ["US","India"]:
  x=clean[clean.country.eq(country)];n=len(x);aux=sum(aux_clean[country,s] for s in [2,3]);rows.append({"population":f"clean {country}","s1":n,"aux_records":aux,"aux_per_s1":aux/n,"blocker_per_s1":x.blocker_pairs.sum()/n,"shortlist_per_s1":x.shortlist_pairs.sum()/n,"true_matches_per_s1":x.true_matches.sum()/n,"accepted_per_s1":x.predicted_edges.sum()/n,"true_matched_aux_fraction":x.true_matches.sum()/aux,"predicted_assigned_fraction":np.nan})
 for country in ["US","India","France"]:
  n=s1_test[country];aux=sum(aux_test[country,s] for s in [2,3]);
  # Existing P002 artifacts only.
  bg=str(P002/f"cache/lightgbm/{country}/s*/*.parquet");sg=str(P002/f"cache/shortlist/{country}/*.parquet")
  c=duckdb.connect();bp=c.execute(f"select count(*)from read_parquet('{bg}',union_by_name=true)").fetchone()[0];sp=c.execute(f"select count(*)from read_parquet('{sg}',union_by_name=true)").fetchone()[0];c.close();acc={"US":2144142,"India":2496737,"France":686518}[country]
  rows.append({"population":f"test {country}","s1":n,"aux_records":aux,"aux_per_s1":aux/n,"blocker_per_s1":bp/n,"shortlist_per_s1":sp/n,"true_matches_per_s1":np.nan,"accepted_per_s1":acc/n,"true_matched_aux_fraction":np.nan,"predicted_assigned_fraction":acc/aux})
 pd.DataFrame(rows).to_csv(OUT/"output/structural_shift.csv",index=False)
 # Source-country clean rates and corresponding test counterfactual.
 dbc=duckdb.connect(str(DB),read_only=True);ids=pd.read_csv(OUT/"clean_eval_ids.tsv",sep="\t");dbc.register("ids",ids[["source1_entity_id"]]);truth=dbc.execute("select q.country,e.target_source,count(*)n from edges e join train_s1 q on q.entity_id=e.s1 join ids i on i.source1_entity_id=e.s1 group by 1,2").fetchdf();dbc.close();cf=[]
 for country in ["US","India"]:
  for source in [2,3]:
   tr=int(truth[(truth.country==country)&(truth.target_source==source)].n.iloc[0]);rate=tr/aux_clean[country,source];test_aux=aux_test[country,source];cf.append({"country":country,"source":f"S{source}","clean_true_edges":tr,"clean_aux_records":aux_clean[country,source],"clean_matched_fraction":rate,"test_aux_records":test_aux,"counterfactual_test_true_edges":rate*test_aux,"counterfactual_matches_per_test_s1":rate*test_aux/s1_test[country]})
 # France has no source-country labeled rate; use source-specific pooled clean US+India rates and flag OOD.
 for source in [2,3]:
  tr=int(truth[truth.target_source==source].n.sum());den=sum(aux_clean[c,source] for c in ["US","India"]);rate=tr/den;cf.append({"country":"France","source":f"S{source}","clean_true_edges":tr,"clean_aux_records":den,"clean_matched_fraction":rate,"test_aux_records":aux_test["France",source],"counterfactual_test_true_edges":rate*aux_test["France",source],"counterfactual_matches_per_test_s1":rate*aux_test["France",source]/s1_test["France"],"confidence":"OOD / pooled source rate"})
 pd.DataFrame(cf).to_csv(OUT/"output/counterfactual_match_rates.csv",index=False);atomic_json(OUT/"runtime/structural.json",{"seconds":time.time()-start,"status":"complete"})

def sanity():
 start=time.time();rows=[];con=duckdb.connect();con.execute("pragma threads=20")
 # Retrieval top-1 distributions and vector incidence from complete blocker caches.
 for pop,country,glob,idcol in [("clean US","US",str(OUT/"cache/lightgbm/US/s*/*.parquet"),"s1"),("clean India","India",str(OUT/"cache/lightgbm/India/s*/*.parquet"),"s1")]+[(f"test {c}",c,str(P002/f"cache/lightgbm/{c}/s*/*.parquet"),"s1") for c in ["US","India","France"]]:
  z=con.execute(f"select {idcol} s1,max(name_retrieval_score)n,max(address_retrieval_score)a from read_parquet('{glob}',union_by_name=true)group by 1").fetchdf()
  rows.append({"population":pop,"metric":"name_retrieval_top1","mean":z.n.mean(),"p50":z.n.quantile(.5),"p90":z.n.quantile(.9),"zero_fraction":z.n.eq(0).mean(),"near_zero_le_0.01_fraction":z.n.le(.01).mean()})
  rows.append({"population":pop,"metric":"address_retrieval_top1","mean":z.a.mean(),"p50":z.a.quantile(.5),"p90":z.a.quantile(.9),"zero_fraction":z.a.eq(0).mean(),"near_zero_le_0.01_fraction":z.a.le(.01).mean()})
 # Shortlist-level missingness/nonfinite and raw length/numeric summaries.
 for pop,glob in [("clean US",str(OUT/"cache/shortlist/US/*.parquet")),("clean India",str(OUT/"cache/shortlist/India/*.parquet"))]+[(f"test {c}",str(P002/f"cache/shortlist/{c}/*.parquet")) for c in ["US","India","France"]]:
  q=con.execute(f"select avg(length(s1_business_name))name_len,avg(length(s1_business_address))addr_len,avg((candidate_business_address='')::int)cand_addr_empty,sum((not isfinite(lightgbm_score))::int)nonfinite,count(*)n from read_parquet('{glob}',union_by_name=true)").fetchdf().iloc[0]
  rows.append({"population":pop,"metric":"shortlist_input","mean_name_length":q.name_len,"mean_address_length":q.addr_len,"empty_candidate_address_fraction":q.cand_addr_empty,"nonfinite_lightgbm_score_count":int(q.nonfinite),"pairs":int(q.n)})
 con.close()
 # Deterministic 100k-pair token-length samples per population.
 tok=AutoTokenizer.from_pretrained(E005/"model/best_checkpoint",use_fast=True,local_files_only=True)
 sys.path.insert(0,str(E005/"src"));from common import serialize
 samples=[]
 for pop,glob,country in [("clean US",str(OUT/"cache/shortlist/US/*.parquet"),"US"),("clean India",str(OUT/"cache/shortlist/India/*.parquet"),"India")]+[(f"test {c}",str(P002/f"cache/shortlist/{c}/*.parquet"),c) for c in ["US","India","France"]]:
  c=duckdb.connect();d=c.execute(f"select s1_business_name,s1_business_address,candidate_business_name,candidate_business_address from read_parquet('{glob}',union_by_name=true) using sample 100000 rows(reservoir,20260925)").fetchdf();c.close();lens=[]
  for i in range(0,len(d),4096):
   x=d.iloc[i:i+4096];a=[serialize(n,z) for n,z in zip(x.s1_business_name,x.s1_business_address)];b=[serialize(n,z) for n,z in zip(x.candidate_business_name,x.candidate_business_address)];la=tok(a,add_special_tokens=False,truncation=False,return_length=True)["length"];lb=tok(b,add_special_tokens=False,truncation=False,return_length=True)["length"];lens.extend(np.asarray(la)+np.asarray(lb)+4)
  lens=np.asarray(lens);samples.append({"population":pop,"sample_n":len(lens),"token_length_mean":lens.mean(),"token_length_p50":np.quantile(lens,.5),"token_length_p90":np.quantile(lens,.9),"token_length_p99":np.quantile(lens,.99),"truncation_fraction_gt104":(lens>104).mean()})
 pd.DataFrame(rows).to_csv(OUT/"output/france_sanity.csv",index=False);pd.DataFrame(samples).to_csv(OUT/"output/token_length_sample.csv",index=False)
 atomic_json(OUT/"runtime/sanity.json",{"seconds":time.time()-start,"status":"complete","token_sample":"deterministic DuckDB reservoir, 100000 pairs/population"})

def scenarios():
 slices=pd.read_csv(OUT/"output/slice_metrics.csv").set_index("slice");f_us=slices.loc["US","macro_f05"];f_in=slices.loc["India","macro_f05"]
 counts={"US":663106,"India":809986,"France":259452};tot=sum(counts.values());w={k:v/tot for k,v in counts.items()};fr=(.937-w["US"]*f_us-w["India"]*f_in)/w["France"]
 ui=(counts["US"]*f_us+counts["India"]*f_in)/(counts["US"]+counts["India"]);degrade=ui-.937
 cf=pd.read_csv(OUT/"output/counterfactual_match_rates.csv");true=float(cf.counterfactual_test_true_edges.sum());pred=5327397;ideal=f05_ideal_precision(pred,true)
 atomic_json(OUT/"output/public_score_scenarios.json",{"public_score":.937,"A_representative_full_test_mix":{"assumption":"public country mix equals full test; US/India equal clean country F0.5","implied_france_f05":fr,"weights":w},"B_no_france_degradation":{"assumption":"France equals S1-weighted clean US/India; all populations share equal additive degradation","clean_ui_f05":ui,"required_joint_additive_degradation":degrade},"C_cardinality_only_ideal_precision":{"assumption":"counterfactual expected true links are truth; every accepted P002 edge is correct; discrepancy is false negatives only","accepted_edges":pred,"counterfactual_true_edges":true,"implied_recall":min(pred/true,1),"ideal_precision_f05":ideal}})

def main():
 total=time.time();score_and_calibration();structural();sanity();scenarios();atomic_json(OUT/"runtime/analysis_total.json",{"seconds":time.time()-total,"status":"complete"});print("ANALYSIS_COMPLETE",time.time()-total)
if __name__=="__main__":main()
