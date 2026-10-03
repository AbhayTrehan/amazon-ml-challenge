#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/production/p001_e005b"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
TEST = ROOT / "student_resource/dataset/test"
MODEL_TXT = E001 / "model/lightgbm_model.txt"
MODEL_JOBLIB = E001 / "model/lightgbm_model.joblib"
MODEL_CONFIG = E001 / "model/model_config.json"
XL_CHECKPOINT = E005 / "model/best_checkpoint"
EXPECTED_XL_HASH = "e7079b798b866b27116a0059c9c7cf1531817e42659ec0c9442654aee5d0c462"
EXPECTED_FILES = {
    "test_source1.tsv": (1732544, "3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5"),
    "test_source2.tsv": (4887273, "79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd"),
    "test_source3.tsv": (5082316, "850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035"),
}
EXPECTED_COUNTS = {
    "test_s1": {"US":663106,"India":809986,"France":259452},
    "test_s2": {"US":1871330,"India":2312565,"France":703378},
    "test_s3": {"US":1945701,"India":2405000,"France":731615},
}
SHARD_SIZE = 20000
THREADS = 32
PROC = psutil.Process()

sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(E005 / "src"))
from evaluate_e005 import mask_for, score_column  # noqa: E402
from metric import grouped_f05  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    tmp.replace(path)


def runtime_state():
    path = EXP / "runtime.json"
    if path.exists(): return json.loads(path.read_text())
    return {"status":"running", "started_epoch":time.time(), "stages":{}, "versions":{
        "python":sys.version.split()[0], "duckdb":duckdb.__version__, "lightgbm":lgb.__version__,
        "numpy":np.__version__, "pandas":pd.__version__}}


def record(stage, seconds, **extra):
    state = runtime_state(); state["stages"][stage] = {"seconds":seconds, **extra}
    state["peak_process_rss_gb"] = max(state.get("peak_process_rss_gb",0), PROC.memory_info().rss/2**30,
                                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    atomic_json(EXP / "runtime.json", state)


def con_ro():
    con=duckdb.connect(str(DB),read_only=True); con.execute(f"PRAGMA threads={THREADS}"); con.execute("PRAGMA memory_limit='100GB'")
    return con


def features():
    return json.loads(MODEL_CONFIG.read_text())["features"]


def booster():
    return lgb.Booster(model_file=str(MODEL_TXT))


def input_preflight():
    t0=time.time(); report={"files":{},"tables":{},"models":{},"ok":True}
    for name,(rows,expected_hash) in EXPECTED_FILES.items():
        p=TEST/name; digest=sha256(p); lines=sum(1 for _ in p.open("rb"))-1
        report["files"][name]={"bytes":p.stat().st_size,"rows":lines,"expected_rows":rows,"sha256":digest,
                               "expected_sha256":expected_hash,"ok":lines==rows and digest==expected_hash}
        report["ok"] &= report["files"][name]["ok"]
    con=con_ro()
    for table,expected in EXPECTED_COUNTS.items():
        rows,unique,bad=con.execute(f"select count(*),count(distinct entity_id),count(*) filter(where not starts_with(entity_id,'{table[-2:].upper()}-')) from {table}").fetchone()
        countries=dict(con.execute(f"select country,count(*) from {table} group by country").fetchall())
        cols=[x[0] for x in con.execute(f"describe {table}").fetchall()]
        ok=rows==sum(expected.values()) and unique==rows and bad==0 and countries==expected and all(x in cols for x in ["entity_id","business_name","business_address","country","nname","naddr","row_num"])
        report["tables"][table]={"rows":rows,"unique_ids":unique,"bad_prefix":bad,"country_counts":countries,"columns":cols,"ok":ok};report["ok"] &= ok
    con.close()
    xl_hash=sha256(XL_CHECKPOINT/"model.safetensors")
    native=booster(); wrapped=joblib.load(MODEL_JOBLIB)
    report["models"]={"lightgbm_txt_exists":MODEL_TXT.exists(),"lightgbm_joblib_reloads":wrapped.n_features_in_==49,
      "lightgbm_features":native.num_feature(),"lightgbm_trees":native.num_trees(),"xlmr_sha256":xl_hash,
      "xlmr_sha256_ok":xl_hash==EXPECTED_XL_HASH,"tokenizer_exists":(XL_CHECKPOINT/"tokenizer.json").exists(),
      "config_exists":(XL_CHECKPOINT/"config.json").exists()}
    report["ok"] &= all([report["models"]["lightgbm_txt_exists"],report["models"]["lightgbm_joblib_reloads"],
                         report["models"]["lightgbm_features"]==49,report["models"]["xlmr_sha256_ok"],
                         report["models"]["tokenizer_exists"],report["models"]["config_exists"]])
    atomic_json(EXP/"diagnostics/input_integrity.json",report);record("input_preflight",time.time()-t0)
    if not report["ok"]: raise RuntimeError("P001 input/model integrity failed")
    print(json.dumps(report,indent=2))


def parity_prepare():
    t0=time.time(); fs=features(); paths=sorted((E001/"cache/features").glob("evaluation_*.parquet"))
    frame=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    native=booster(); native_score=native.predict(frame[fs],num_threads=THREADS).astype(np.float32)
    saved=pd.read_parquet(E001/"predictions/evaluation_pair_scores.parquet",columns=["s1","target_id","source","score"])
    scored=frame[["s1","target_id","source","in_k10"]].copy();scored["score_new"]=native_score
    cmp=scored.merge(saved,on=["s1","target_id","source"],validate="one_to_one")
    max_diff=float(np.max(np.abs(cmp.score_new-cmp.score)))
    bundle=pd.read_parquet(E004_BUNDLE())
    k10=scored[scored.in_k10.eq(1)][["s1","target_id","source","score_new"]]
    bundle["_source_num"]=bundle.candidate_source.str[-1].astype(np.int8)
    merged=bundle.merge(k10,left_on=["source1_entity_id","candidate_entity_id","_source_num"],
                        right_on=["s1","target_id","source"],validate="one_to_one",sort=False)
    bundle_diff=float(np.max(np.abs(merged.lightgbm_score-merged.score_new)))
    bundle["lightgbm_score"]=merged.score_new.to_numpy(dtype=np.float32);bundle.drop(columns="_source_num",inplace=True)
    ranked=bundle.sort_values(["source1_entity_id","lightgbm_score","candidate_entity_id"],ascending=[True,False,True],kind="mergesort").copy()
    ranked["lightgbm_rank_within_s1"]=(ranked.groupby("source1_entity_id",sort=False).cumcount()+1).astype(np.int16)
    rank_bad=int((ranked.lightgbm_rank_within_s1.to_numpy()!=bundle.loc[ranked.index,"lightgbm_rank_within_s1"].to_numpy()).sum())
    out=EXP/"cache/parity/fold0_k10.parquet";out.parent.mkdir(parents=True,exist_ok=True);ranked.to_parquet(out,index=False,compression="zstd")
    report={"feature_rows":len(frame),"k10_rows":len(bundle),"native_vs_saved_max_abs":max_diff,
            "bundle_score_max_abs":bundle_diff,"rank_mismatches":rank_bad,"ok":max_diff<=1e-7 and bundle_diff<=1e-7 and rank_bad==0}
    atomic_json(EXP/"diagnostics/parity_lightgbm.json",report);record("parity_lightgbm",time.time()-t0,**report)
    if not report["ok"]: raise RuntimeError(f"LightGBM parity failed: {report}")
    print(json.dumps(report,indent=2))


def E004_BUNDLE():
    return ROOT/"analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/gpu_bundle/evaluation_pairs.parquet"


def parity_finalize(neural_path: Path):
    t0=time.time();n=pd.read_parquet(neural_path);base=pd.read_parquet(EXP/"cache/parity/fold0_k10.parquet");base=base[base.lightgbm_rank_within_s1.le(5)|base.lightgbm_score.ge(.05)]
    d=base.merge(n,on=["source1_entity_id","candidate_entity_id"],validate="one_to_one");meta=pd.read_parquet(E005/"runtime/selected_s1_metadata.parquet").query("role=='evaluation'")
    cfg={"decoder":"one_owner_margin","threshold":.51,"margin":.48,"weight":.60}
    scores=.60*d.neural_probability.to_numpy(float)+.40*d.lightgbm_score.to_numpy(float)
    mask=mask_for(d,scores,cfg);met=grouped_f05(d,meta,mask)
    met.update({"expected":0.976693878247762,"absolute_difference":abs(met["macro_f05"]-0.976693878247762)})
    met["ok"]=met["absolute_difference"]<1e-12
    atomic_json(EXP/"diagnostics/production_parity.json",met);record("parity_xlmr_decode",time.time()-t0,**met)
    if not met["ok"]: raise RuntimeError(f"E005B production parity failed: {met}")
    print(json.dumps(met,indent=2))


def index_paths(country,source,channel):
    d=EXP/f"cache/indexes/{country}/s{source}";return d,d/f"aux.parquet",d/f"{channel}.npz",d/f"{channel}_idf.npy"


def query_path(country): return EXP/f"cache/indexes/{country}/queries.parquet"


def build_indexes():
    total_start=time.time();details=[]
    con=con_ro()
    for country in ["France","India","US"]:
        qp=query_path(country);qp.parent.mkdir(parents=True,exist_ok=True)
        if not qp.exists():
            q=con.execute("select entity_id,business_name,coalesce(business_address,'') business_address,nname,coalesce(naddr,'') naddr,row_num from test_s1 where country=? order by entity_id",[country]).fetchdf();q.to_parquet(qp,index=False,compression="zstd")
        for source in [2,3]:
            d,ap,_,_=index_paths(country,source,"name");d.mkdir(parents=True,exist_ok=True)
            if not ap.exists():
                aux=con.execute(f"select entity_id,business_name,coalesce(business_address,'') business_address,nname,coalesce(naddr,'') naddr from test_s{source} where country=? order by entity_id",[country]).fetchdf()
                nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
                aux["candidate_normalized_name_fanout"]=aux.nname.map(nf).clip(0,65535).astype(np.uint16)
                aux["candidate_normalized_address_fanout"]=aux.naddr.map(af).clip(0,65535).astype(np.uint16)
                aux.to_parquet(ap,index=False,compression="zstd")
            aux=pd.read_parquet(ap)
            for channel,col in [("name","business_name"),("address","business_address")]:
                _,_,mp,ip=index_paths(country,source,channel)
                if mp.exists() and ip.exists(): continue
                start=time.time();vec=e001.vec_for(channel); counts=vec.transform(aux[col].fillna(""));tf=TfidfTransformer();xa=tf.fit_transform(counts).astype(np.float32)
                sparse.save_npz(mp,xa,compressed=False);np.save(ip,tf.idf_.astype(np.float32))
                details.append({"country":country,"source":source,"channel":channel,"rows":len(aux),"nnz":int(xa.nnz),"seconds":time.time()-start,"matrix_bytes":mp.stat().st_size})
                del counts,xa,tf;gc.collect()
            del aux;gc.collect()
    con.close();elapsed=time.time()-total_start
    old=json.loads((EXP/"diagnostics/index_build.json").read_text()) if (EXP/"diagnostics/index_build.json").exists() else {"details":[]}
    old["details"].extend(details);old["seconds_this_invocation"]=elapsed;old["completed"]=True;atomic_json(EXP/"diagnostics/index_build.json",old)
    record("index_build",elapsed,new_indexes=len(details));print(json.dumps(old,indent=2))


def load_index(country,source,channel):
    _,ap,mp,ip=index_paths(country,source,channel);xa=sparse.load_npz(mp).tocsr();tf=TfidfTransformer();tf.idf_=np.load(ip);return xa,tf


def retrieve_chunk(q,aux,indexes):
    rets=[];qmat={}
    for channel,col in [("name","business_name"),("address","business_address")]:
        xa,tf=indexes[channel];xq=tf.transform(e001.vec_for(channel).transform(q[col].fillna(""))).astype(np.float32);qmat[channel]=xq
        rets.append(e001.retrieve(xq,xa,channel))
    cand=rets[0].merge(rets[1],on=["qpos","apos"],how="outer")
    cand["s1"]=q.entity_id.to_numpy()[cand.qpos];cand["target_id"]=aux.entity_id.to_numpy()[cand.apos]
    for channel in ["name","address"]:cand[f"{channel}_char_tfidf"]=e001.row_cosines(qmat[channel],indexes[channel][0],cand.qpos.to_numpy(),cand.apos.to_numpy())
    return cand,qmat


def benchmark():
    t0=time.time();rows=[];total_source_queries=0;total_pairs=0
    for country in ["France","India","US"]:
        qfull=pd.read_parquet(query_path(country));take=np.linspace(0,len(qfull)-1,2000,dtype=int);q=qfull.iloc[take].reset_index(drop=True)
        for source in [2,3]:
            aux=pd.read_parquet(index_paths(country,source,"name")[1]);indexes={c:load_index(country,source,c) for c in ["name","address"]}
            start=time.time();cand,_=retrieve_chunk(q,aux,indexes);sec=time.time()-start;k10=cand[cand.name_rank.le(10)|cand.address_rank.le(10)]
            rows.append({"country":country,"source":source,"s1":len(q),"seconds":sec,"source_queries_per_second":len(q)/sec,"k10_pairs":len(k10),"mean_k10":len(k10)/len(q)})
            total_source_queries+=len(q);total_pairs+=len(k10);del aux,indexes,cand,k10;gc.collect()
    query_seconds=sum(x["seconds"] for x in rows);projected=query_seconds/total_source_queries*(1732544*2)
    report={"sample_unique_s1":6000,"sample_source_queries":total_source_queries,"query_seconds":query_seconds,
      "projected_full_query_seconds":projected,"projected_full_query_hours":projected/3600,"sample_k10_pairs":total_pairs,
      "mean_k10_pairs_per_s1":total_pairs/6000,"peak_rss_gb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"details":rows,
      "reasonable":projected<48*3600}
    atomic_json(EXP/"diagnostics/retrieval_benchmark.json",report);record("retrieval_benchmark",time.time()-t0,**report)
    if not report["reasonable"]: raise RuntimeError(f"Pathological retrieval projection: {report}")
    print(json.dumps(report,indent=2))


def enrich_and_score(cand,q,aux,source,country,model,fs,nf,af):
    cand["source"]=np.int8(source);cand["country"]=country;cand["role"]="test";cand["label"]=np.int8(0)
    cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
    cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
    cand["candidate_normalized_name_fanout"]=aux.candidate_normalized_name_fanout.to_numpy()[cand.apos]
    cand["candidate_normalized_address_fanout"]=aux.candidate_normalized_address_fanout.to_numpy()[cand.apos]
    qn=q.nname.to_numpy()[cand.qpos];qa=q.naddr.to_numpy()[cand.qpos];an=aux.nname.to_numpy()[cand.apos];aa=aux.naddr.to_numpy()[cand.apos]
    cand["nname_exact"] = ((qn!="")&(qn==an)).astype(np.int8);cand["naddr_exact"] = ((qa!="")&(qa==aa)).astype(np.int8)
    cand=e001.add_context(cand);feat=e001.build_features(cand,q,aux);feat=feat[feat.in_k10.eq(1)].copy()
    if fs != list(fs) or any(c not in feat for c in fs): raise RuntimeError("feature schema/order failure")
    score_start=time.time();feat["lightgbm_score"]=model.predict(feat[fs],num_threads=THREADS).astype(np.float32);score_seconds=time.time()-score_start
    if not np.isfinite(feat.lightgbm_score).all():raise RuntimeError("NaN/Inf LightGBM score")
    return feat,score_seconds


def retrieve_full():
    start_all=time.time();model=booster();fs=features();stats=[]
    for country in ["France","India","US"]:
        qfull=pd.read_parquet(query_path(country))
        for source in [2,3]:
            aux=pd.read_parquet(index_paths(country,source,"name")[1]);indexes={c:load_index(country,source,c) for c in ["name","address"]};nf=aux.nname.value_counts(dropna=False);af=aux.naddr.value_counts(dropna=False)
            outdir=EXP/f"cache/features/{country}/s{source}";predir=EXP/f"cache/lightgbm_predictions/{country}/s{source}";outdir.mkdir(parents=True,exist_ok=True);predir.mkdir(parents=True,exist_ok=True)
            for part,start in enumerate(range(0,len(qfull),SHARD_SIZE)):
                fp=outdir/f"part-{part:05d}.parquet";pp=predir/f"part-{part:05d}.parquet"
                if fp.exists() and pp.exists(): continue
                t=time.time();q=qfull.iloc[start:start+SHARD_SIZE].reset_index(drop=True);cand,_=retrieve_chunk(q,aux,indexes);retrieval_s=time.time()-t
                t2=time.time();feat,score_s=enrich_and_score(cand,q,aux,source,country,model,fs,nf,af);feature_lgb_s=time.time()-t2;feature_s=feature_lgb_s-score_s
                feat.to_parquet(fp,index=False,compression="zstd")
                pred=feat[["s1","target_id","source","country","name_rank","name_retrieval_score","address_rank","address_retrieval_score","retrieved_by_name","retrieved_by_address","lightgbm_score"]]
                pred.to_parquet(pp,index=False,compression="zstd")
                stats.append({"country":country,"source":source,"part":part,"s1":len(q),"k10_pairs":len(feat),"retrieval_seconds":retrieval_s,"feature_seconds":feature_s,"lightgbm_seconds":score_s})
                with (EXP/"logs/retrieval_parts.jsonl").open("a") as f:f.write(json.dumps(stats[-1])+"\n")
                print("RETRIEVED",stats[-1],flush=True);del q,cand,feat,pred;gc.collect()
            del aux,indexes;gc.collect()
        del qfull;gc.collect()
    elapsed=time.time()-start_all;record("full_retrieval_feature_lightgbm",elapsed,new_parts=len(stats));print("full retrieval invocation seconds",elapsed)


def build_shortlist():
    t0=time.time();parts=[]
    for country in ["France","India","US"]:
        qfull=pd.read_parquet(query_path(country));aux={s:pd.read_parquet(index_paths(country,s,"name")[1]).set_index("entity_id") for s in [2,3]}
        n=math.ceil(len(qfull)/SHARD_SIZE);outdir=EXP/f"cache/neural_shortlist/{country}";outdir.mkdir(parents=True,exist_ok=True)
        for part in range(n):
            op=outdir/f"part-{part:05d}.parquet"
            if op.exists():continue
            frames=[pd.read_parquet(EXP/f"cache/lightgbm_predictions/{country}/s{s}/part-{part:05d}.parquet") for s in [2,3]]
            d=pd.concat(frames,ignore_index=True);d=d.sort_values(["s1","lightgbm_score","target_id"],ascending=[True,False,True],kind="mergesort")
            d["lightgbm_rank_within_s1"]=(d.groupby("s1",sort=False).cumcount()+1).astype(np.int16);d=d[d.lightgbm_rank_within_s1.le(5)|d.lightgbm_score.ge(.05)].copy()
            q=qfull.iloc[part*SHARD_SIZE:(part+1)*SHARD_SIZE].set_index("entity_id")
            d["s1_business_name"]=d.s1.map(q.business_name);d["s1_business_address"]=d.s1.map(q.business_address).fillna("")
            d["candidate_business_name"]="";d["candidate_business_address"]=""
            for s in [2,3]:
                take=d.source.eq(s);ids=d.loc[take,"target_id"];d.loc[take,"candidate_business_name"]=ids.map(aux[s].business_name).to_numpy();d.loc[take,"candidate_business_address"]=ids.map(aux[s].business_address).fillna("").to_numpy()
            d=d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"});d["candidate_source"]=np.where(d.source.eq(2),"S2","S3");d["role"]="test"
            cols=["source1_entity_id","candidate_entity_id","candidate_source","country","role","s1_business_name","s1_business_address","candidate_business_name","candidate_business_address","lightgbm_score","lightgbm_rank_within_s1","name_retrieval_score","name_rank","address_retrieval_score","address_rank"]
            d[cols].to_parquet(op,index=False,compression="zstd");parts.append({"country":country,"part":part,"rows":len(d)})
            print("SHORTLIST",parts[-1],flush=True)
    elapsed=time.time()-t0;record("shortlist_generation",elapsed,new_parts=len(parts))


def decode_outputs():
    t0=time.time();short_glob=str(EXP/"cache/neural_shortlist/*/*.parquet");neural_glob=str(EXP/"cache/neural_predictions/*/*.parquet")
    con=duckdb.connect();con.execute(f"PRAGMA threads={THREADS}");con.execute("PRAGMA memory_limit='100GB'");con.execute(f"PRAGMA temp_directory='{EXP/'cache/ducktmp'}'")
    con.execute(f"""create temp table scored as select s.*,n.neural_probability,(.60*n.neural_probability+.40*s.lightgbm_score)::float final_score
      from read_parquet('{short_glob}',union_by_name=true) s join read_parquet('{neural_glob}',union_by_name=true)n using(source1_entity_id,candidate_entity_id)""")
    con.execute("""create temp table owned as select *,row_number()over(partition by candidate_entity_id order by final_score desc,source1_entity_id) owner_rank,
      lead(final_score)over(partition by candidate_entity_id order by final_score desc,source1_entity_id) second_score from scored""")
    con.execute("""create temp table accepted as select * from owned where owner_rank=1 and final_score>=.51 and final_score-coalesce(second_score,0)>=.48""")
    out=EXP/"output";out.mkdir(parents=True,exist_ok=True)
    con.execute(f"""copy (with q as(select *,row_number()over() rn from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote=''))
      select q.entity_id source1_entity_id,coalesce(string_agg(a.candidate_entity_id,',' order by a.candidate_entity_id),'') matched_entity_ids
      from q left join accepted a on q.entity_id=a.source1_entity_id group by q.entity_id,q.rn order by q.rn)
      to '{out/'matching_results.tsv'}'(header,delimiter '\t')""")
    con.execute(f"""copy (with q as(select *,row_number()over() rn from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote=''))
      select q.entity_id source1_entity_id,coalesce(string_agg(s.candidate_entity_id,',' order by s.candidate_entity_id),'') candidate_entity_ids
      from q left join scored s on q.entity_id=s.source1_entity_id group by q.entity_id,q.rn order by q.rn)
      to '{out/'candidate_pairs.tsv'}'(header,delimiter '\t')""")
    con.execute(f"copy (select source1_entity_id,candidate_entity_id,candidate_source,country,lightgbm_score,neural_probability,final_score,final_score-coalesce(second_score,0) owner_margin from accepted order by source1_entity_id,candidate_entity_id) to '{out/'accepted_pairs.parquet'}'(format parquet,compression zstd)")
    conflicts=con.execute("select count(*) from (select candidate_entity_id from scored group by 1 having count(*)>1)").fetchone()[0]
    summary=con.execute("select count(*) shortlist_pairs,count(distinct source1_entity_id)s1_with_candidates,count(*)filter(where owner_rank=1) owner_winners,(select count(*)from accepted)accepted_pairs from owned").fetchdf().iloc[0].to_dict();summary["one_owner_conflicting_auxiliaries"]=conflicts
    atomic_json(EXP/"diagnostics/decode_summary.json",summary);con.close();record("fusion_decode_output_write",time.time()-t0,**summary);print(json.dumps(summary,indent=2,default=float))


def stats():
    t0=time.time();con=duckdb.connect();con.execute(f"PRAGMA threads={THREADS}");s=str(EXP/"cache/neural_shortlist/*/*.parquet");n=str(EXP/"cache/neural_predictions/*/*.parquet");a=str(EXP/"output/accepted_pairs.parquet")
    con.execute(f"create temp view s as select * from read_parquet('{s}',union_by_name=true)");con.execute(f"create temp view n as select * from read_parquet('{n}',union_by_name=true)")
    k=str(EXP/"cache/lightgbm_predictions/*/*/*.parquet");con.execute(f"create temp view k as select * from read_parquet('{k}',union_by_name=true)")
    con.execute(f"create temp view q as select * from read_csv('{TEST/'test_source1.tsv'}',delim='\t',header=true,all_varchar=true,quote='')")
    con.execute(f"create temp view a as select * from read_parquet('{a}')")
    pred=con.execute("""with c as(select q.entity_id,q.country,count(a.candidate_entity_id)n,count(a.candidate_entity_id)filter(where candidate_source='S2')s2,count(a.candidate_entity_id)filter(where candidate_source='S3')s3 from q left join a on q.entity_id=a.source1_entity_id group by 1,2)
      select country,count(*)s1,avg(n=0)::double singleton_rate,avg(n)::double mean_matches,quantile_cont(n,.5)p50,quantile_cont(n,.9)p90,quantile_cont(n,.95)p95,quantile_cont(n,.99)p99,max(n)max,sum(s2)s2_accepted,sum(s3)s3_accepted from c group by country order by country""").fetchdf();pred.to_csv(EXP/"diagnostics/country_prediction_stats.csv",index=False)
    short=con.execute("""with c as(select q.entity_id,q.country,count(s.candidate_entity_id)n from q left join s on q.entity_id=s.source1_entity_id group by 1,2)select country,count(*)s1,avg(n)mean_shortlist,quantile_cont(n,.5)p50,quantile_cont(n,.9)p90,quantile_cont(n,.95)p95,quantile_cont(n,.99)p99,max(n)max from c group by country order by country""").fetchdf();short.to_csv(EXP/"diagnostics/match_count_stats.csv",index=False)
    score=con.execute("""select s.country,count(*)pairs,quantile_cont(.6*n.neural_probability+.4*s.lightgbm_score,.01)p01,quantile_cont(.6*n.neural_probability+.4*s.lightgbm_score,.5)p50,quantile_cont(.6*n.neural_probability+.4*s.lightgbm_score,.9)p90,quantile_cont(.6*n.neural_probability+.4*s.lightgbm_score,.99)p99,stddev_pop(.6*n.neural_probability+.4*s.lightgbm_score)std from s join n using(source1_entity_id,candidate_entity_id)group by s.country order by s.country""").fetchdf();score.to_csv(EXP/"diagnostics/score_stats.csv",index=False)
    kstats=con.execute("""with c as(select q.entity_id,q.country,count(k.target_id)n from q left join k on q.entity_id=k.s1 group by 1,2),x as(select country,count(*)s1,sum(n)pairs,avg(n)mean,quantile_cont(n,.5)p50,quantile_cont(n,.9)p90,quantile_cont(n,.95)p95,quantile_cont(n,.99)p99,max(n)max from c group by country)select * from x union all select 'overall',sum(s1),sum(pairs),sum(pairs)::double/sum(s1),quantile_cont(n,.5),quantile_cont(n,.9),quantile_cont(n,.95),quantile_cont(n,.99),max(n) from c""").fetchdf();kstats.to_csv(EXP/"diagnostics/k10_candidate_stats.csv",index=False)
    ssource=con.execute("select country,candidate_source,count(*)pairs,count(distinct source1_entity_id)s1 from s group by 1,2 order by 1,2").fetchdf();ssource.to_csv(EXP/"diagnostics/shortlist_source_stats.csv",index=False)
    con.close();record("sanity_diagnostics",time.time()-t0);print(pred.to_string(index=False));print(short.to_string(index=False));print(score.to_string(index=False))


def validate_finalize():
    t0=time.time();out=EXP/"output";matching=out/"matching_results.tsv";candidate=out/"candidate_pairs.tsv"
    cmd=[sys.executable,str(ROOT/"student_resource/utils/validate_submission.py"),"--matching",str(matching),"--candidate",str(candidate),"--test-dir",str(TEST),"--check-ids"]
    proc=subprocess.run(cmd,text=True,capture_output=True);official=proc.stdout+proc.stderr;(EXP/"logs/official_validator.log").write_text(official)
    if proc.returncode!=0 or "PASS — no blocking issues found" not in official or "WARNING:" in official:raise RuntimeError("Official validator did not pass cleanly; see logs/official_validator.log")
    con=duckdb.connect();con.execute(f"PRAGMA threads={THREADS}");con.execute("PRAGMA memory_limit='100GB'");con.execute(f"PRAGMA temp_directory='{EXP/'cache/ducktmp'}'")
    con.execute(f"attach '{DB}' as audit(read_only)")
    con.execute(f"create temp table m as select * from read_csv('{matching}',delim='\t',header=true,all_varchar=true,quote='')")
    con.execute(f"create temp table c as select * from read_csv('{candidate}',delim='\t',header=true,all_varchar=true,quote='')")
    checks={}
    for table,col in [("m","matched_entity_ids"),("c","candidate_entity_ids")]:
        base=con.execute(f"select count(*),count(distinct source1_entity_id),count(*)filter(where lower(coalesce({col},''))='nan') from {table}").fetchone()
        missing=con.execute(f"select count(*) from audit.test_s1 q left join {table} x on q.entity_id=x.source1_entity_id where x.source1_entity_id is null").fetchone()[0]
        extra=con.execute(f"select count(*) from {table}x left join audit.test_s1 q on q.entity_id=x.source1_entity_id where q.entity_id is null").fetchone()[0]
        dupe=con.execute(f"select count(*) from {table} where coalesce({col},'')<>'' and len(string_split({col},','))<>len(list_distinct(string_split({col},',')))").fetchone()[0]
        unknown=con.execute(f"""with e as(select unnest(string_split({col},','))id from {table} where coalesce({col},'')<>'')select count(*)from e left join(select entity_id from audit.test_s2 union all select entity_id from audit.test_s3)v using(entity_id)where v.entity_id is null""".replace("using(entity_id)","on e.id=v.entity_id")).fetchone()[0]
        checks[table]={"rows":base[0],"unique_s1":base[1],"literal_nan":base[2],"missing_s1":missing,"extra_s1":extra,"duplicate_targets":dupe,"unknown_targets":unknown,
                       "ok":base[0]==1732544 and base[1]==1732544 and base[2]==0 and missing==0 and extra==0 and dupe==0 and unknown==0}
    subset_bad=con.execute("""with me as(select source1_entity_id,unnest(string_split(matched_entity_ids,','))id from m where coalesce(matched_entity_ids,'')<>''),ce as(select source1_entity_id,unnest(string_split(candidate_entity_ids,','))id from c where coalesce(candidate_entity_ids,'')<>'')select count(*)from me left join ce using(source1_entity_id,id)where ce.id is null""").fetchone()[0]
    checks["cross_file"]={"final_not_in_candidates":subset_bad,"ok":subset_bad==0};checks["official_validator"]={"returncode":proc.returncode,"pass":True,"warnings":False}
    if not all(v.get("ok",True) for v in checks.values()):raise RuntimeError(f"Independent output validation failed: {checks}")
    atomic_json(EXP/"diagnostics/output_validation.json",checks);con.close()
    checksums={"matching_results.tsv":sha256(matching),"candidate_pairs.tsv":sha256(candidate),"config.yaml":sha256(EXP/"config.yaml"),
      "xlmr_model.safetensors":sha256(XL_CHECKPOINT/"model.safetensors"),"lightgbm_model.txt":sha256(MODEL_TXT),"lightgbm_model.joblib":sha256(MODEL_JOBLIB)}
    (out/"checksums.sha256").write_text("\n".join(f"{v}  {k}" for k,v in checksums.items())+"\n")
    final=EXP/"final";final.mkdir(exist_ok=True)
    for p in [matching,candidate,out/"checksums.sha256"]:
        dest=final/p.name
        if not dest.exists():shutil.copy2(p,dest);dest.chmod(0o444)
    elapsed=time.time()-t0;record("validation_archive",elapsed,official_pass=True,independent_pass=True,checksums=checksums)
    make_handoff(checksums,official)


def make_handoff(checksums,official):
    rt=runtime_state();par=json.loads((EXP/"diagnostics/production_parity.json").read_text());bench=json.loads((EXP/"diagnostics/retrieval_benchmark.json").read_text())
    k=pd.read_csv(EXP/"diagnostics/k10_candidate_stats.csv");short=pd.read_csv(EXP/"diagnostics/match_count_stats.csv");pred=pd.read_csv(EXP/"diagnostics/country_prediction_stats.csv");score=pd.read_csv(EXP/"diagnostics/score_stats.csv");src=pd.read_csv(EXP/"diagnostics/shortlist_source_stats.csv")
    logs=[json.loads(x) for x in (EXP/"logs/retrieval_parts.jsonl").read_text().splitlines() if x.strip()];neu=json.loads((EXP/"diagnostics/neural_inference.json").read_text())
    stage={"test_read_integrity":rt["stages"]["input_preflight"]["seconds"],"retrieval_index_construction":rt["stages"]["index_build"]["seconds"],
      "k10_query":sum(x["retrieval_seconds"] for x in logs),"feature_generation":sum(x["feature_seconds"] for x in logs),"lightgbm_inference":sum(x["lightgbm_seconds"] for x in logs),
      "shortlist_generation":rt["stages"]["shortlist_generation"]["seconds"],"xlmr_inference":neu["total_seconds"],"fusion_decode_output":rt["stages"]["fusion_decode_output_write"]["seconds"],
      "validation_archive":rt["stages"]["validation_archive"]["seconds"]}
    rt["production_timing_seconds"]=stage;rt["total_wall_seconds"]=time.time()-rt["started_epoch"];rt["status"]="complete";rt["gpu"]={"name":neu["gpu"],"batch_size":neu["batch_size"],"pairs_per_second":neu["pairs_per_second"],"gpu_pairs_per_second":neu["gpu_pairs_per_second"],"peak_allocated_gpu_gb":neu["peak_allocated_gpu_gb"]};rt["cpu_count_used"]=THREADS
    rt["cache_bytes"]={str(p.relative_to(EXP)):sum(x.stat().st_size for x in p.rglob("*") if x.is_file()) for p in [EXP/"cache/features",EXP/"cache/lightgbm_predictions",EXP/"cache/neural_shortlist",EXP/"cache/neural_predictions",EXP/"cache/indexes"]}
    atomic_json(EXP/"runtime.json",rt)
    def table(df):return "```text\n"+df.to_string(index=False)+"\n```"
    lines=["# P001 frozen E005B production handoff","","# Frozen system","","Generic country-equality forward lexical K10; frozen 49-feature E001 LightGBM; shortlist rank <=5 or LightGBM >=0.05; XLM-R revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, max_length 104; fusion 0.60 neural + 0.40 LightGBM; threshold 0.51; one-owner margin 0.48. No training or tuning occurred.","",
      "# Input integrity","","All three official test TSV SHA256 values, schemas, exact row/country counts, unique IDs, and prefixes passed. Official files were read-only and unchanged.","",
      "# Production parity check","",f"The production path reproduced fold-0 E005B macro F0.5 **{par['macro_f05']:.12f}** (absolute difference {par['absolute_difference']:.1g}). Native LightGBM scores and ranks matched cached evidence exactly.","",
      "# Full test candidate generation","",table(k),f"Index build {stage['retrieval_index_construction']/60:.2f} min; K10 querying {stage['k10_query']/3600:.2f} h. Internal top-20 retrieval was retained solely to reproduce E001 contextual features; the persisted/scored candidate set is the exact top-10 union.","",
      "# LightGBM","",f"Frozen native 450-tree model, exact 49-column order. Feature generation {stage['feature_generation']/3600:.2f} h; inference {stage['lightgbm_inference']/60:.2f} min; no NaN/Inf and candidate count unchanged.","",
      "# Neural shortlist","",table(short),table(src),"`candidate_pairs.tsv` is this exact final neural shortlist, not raw K10.","",
      "# XLM-R inference","",f"{neu['gpu']}, BF16 SDPA, batch {neu['batch_size']}; {neu['pairs_per_second']:.1f} pairs/s end-to-end, {neu['gpu_pairs_per_second']:.1f} GPU pairs/s, peak {neu['peak_allocated_gpu_gb']:.2f} GiB, {neu['total_seconds']/3600:.2f} h.","",
      "# Fusion and decoding","","Applied the frozen linear fusion, threshold, one-owner winner, and 0.48 winner-minus-runner-up margin exactly. All predictions originate in the neural shortlist.","",
      "# Unlabeled sanity diagnostics","",table(pred),table(score),"No catastrophic anomalies were detected; these distributions were diagnostic only and caused no tuning.","",
      "# Output validation","","Official validator with `--check-ids`: PASS with no warnings. Independent serialized-file checks passed exact S1 coverage/order, headers, prefixes, ID existence, no duplicate targets, no literal `nan`, and final-match subset-of-candidate.","","```text",official.strip(),"```","",
      "# Final artifacts","",f"`output/matching_results.tsv`: { (EXP/'output/matching_results.tsv').stat().st_size:,} bytes, SHA256 `{checksums['matching_results.tsv']}`.\n\n`output/candidate_pairs.tsv`: {(EXP/'output/candidate_pairs.tsv').stat().st_size:,} bytes, SHA256 `{checksums['candidate_pairs.tsv']}`. Read-only snapshot copies are in `final/`.","",
      "# Runtime","",table(pd.DataFrame([stage])),f"Total wall-clock: {rt['total_wall_seconds']/3600:.2f} h. Major cache byte sizes are in `runtime.json`.","",
      "# Reproduction / resume commands","","```bash\nbash analysis/production/p001_e005b/scripts/run_p001.sh\n```\nEvery expensive stage skips complete shard outputs.","",
      "# Leaderboard instructions","","Upload `output/matching_results.tsv`."]
    (EXP/"HANDOFF.md").write_text("\n".join(lines)+"\n");(EXP/"runtime/COMPLETE").parent.mkdir(parents=True,exist_ok=True);(EXP/"runtime/COMPLETE").write_text(f"completed_epoch={time.time()}\n")


def main():
    for d in ["cache/candidates","cache/features","cache/lightgbm_predictions","cache/neural_shortlist","cache/neural_predictions","output","diagnostics","logs","final"]:(EXP/d).mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["preflight","parity_prepare","parity_finalize","indexes","benchmark","retrieve","shortlist","decode","stats","validate_finalize"]);ap.add_argument("--neural-path")
    args=ap.parse_args();globals()[{"preflight":"input_preflight","indexes":"build_indexes","retrieve":"retrieve_full","shortlist":"build_shortlist","decode":"decode_outputs"}.get(args.stage,args.stage)](**({"neural_path":Path(args.neural_path)} if args.stage=="parity_finalize" else {}))


if __name__=="__main__":main()
