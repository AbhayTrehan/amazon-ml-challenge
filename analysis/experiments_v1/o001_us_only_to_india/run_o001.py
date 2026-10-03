#!/usr/bin/env python3
"""O001 country-held-out diagnostic: frozen standard vs US-only on clean fold-4 India."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/experiments_v1/o001_us_only_to_india"
E001 = ROOT / "analysis/experiments_v1/e001_classical_baseline"
E004 = ROOT / "analysis/experiments_v1/e004_k10_decode_and_gpu_bundle"
E005 = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
R006 = ROOT / "analysis/experiments_v1/r006a_selective_postings"
P002 = ROOT / "analysis/production/p002_final"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
SPLIT = ROOT / "analysis/data_audit_v1/artifacts/proposed_validation_split.tsv"
ASSIGNMENT = E001 / "cache/auxiliary_fold_assignment.parquet"
STD_MODEL = E001 / "model/lightgbm_model.txt"
MODEL_CONFIG = E001 / "model/model_config.json"
SHARD = 20_000
KEY = ["source1_entity_id", "candidate_entity_id", "candidate_source"]

sys.path.insert(0, str(E001 / "scripts"))
import prepare_baseline as e001  # noqa: E402
sys.path.insert(0, str(R006))
import run_r006a, run_r006b  # noqa: E402
sys.path.insert(0, str(P002 / "src"))
from feature_builder import add_context, build_features  # noqa: E402
sys.path.insert(0, str(E005 / "src"))
from metric import grouped_f05, self_test  # noqa: E402
from evaluate_e005 import mask_for, search_policy  # noqa: E402


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def init_dirs() -> None:
    for d in ["cache/index", "cache/blocker/s2", "cache/blocker/s3", "cache/features/s2",
              "cache/features/s3", "cache/calibration", "training", "model/lightgbm",
              "model/xlmr", "predictions", "runtime", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def prepare() -> None:
    """Freeze cohort, prove leakage exclusions, filter training, fit US-only LGBM."""
    t0 = time.time(); init_dirs(); self_test()
    split = pd.read_csv(SPLIT, sep="\t")
    selected = pd.read_parquet(E001 / "selected_ids/selected_s1.parquet")
    hist_cal = selected[selected.role.eq("calibration")].copy()
    if len(hist_cal) != 10_000 or set(hist_cal.fold) != {4}:
        raise RuntimeError("historical fold-4 calibration contract failed")
    ids = split[(split.fold.eq(4)) & split.country.eq("India") &
                ~split.source1_entity_id.isin(hist_cal.source1_entity_id)].copy()
    ids.sort_values("source1_entity_id", kind="mergesort", inplace=True)
    if len(ids) != 173_511 or not ids.source1_entity_id.is_unique:
        raise RuntimeError(f"India clean fold-4 contract failed: {len(ids)}")
    ids.to_csv(OUT / "validation_ids.tsv", sep="\t", index=False)

    con = duckdb.connect(str(DB), read_only=True); con.register("ids", ids[["source1_entity_id"]])
    checks = {
        "lightgbm_training_s1": int(con.execute(
            f"select count(distinct f.s1) from read_parquet('{E001/'cache/features/train_*.parquet'}') f join ids i on i.source1_entity_id=f.s1").fetchone()[0]),
        "xlmr_training_s1": int(con.execute(
            f"select count(distinct f.source1_entity_id) from read_parquet('{E004/'gpu_bundle/train_pairs.parquet'}') f join ids i using(source1_entity_id)").fetchone()[0]),
        "historical_fold4_calibration_s1": int(con.execute(
            "select count(*) from ids i join hist_cal h on false").fetchone()[0]) if False else
            len(set(ids.source1_entity_id) & set(hist_cal.source1_entity_id)),
    }
    con.close()
    if any(checks.values()): raise RuntimeError(f"validation leakage: {checks}")

    train_paths = sorted((E001 / "cache/features").glob("train_*.parquet"))
    train = pd.concat([pd.read_parquet(p) for p in train_paths], ignore_index=True)
    rows_before = len(train); us = train[train.country.eq("US")].copy()
    cfg_all = json.loads(MODEL_CONFIG.read_text()); features = cfg_all["features"]
    cfg = cfg_all["config"]
    model = lgb.LGBMClassifier(**cfg); model.fit(us[features], us.label)
    model.booster_.save_model(str(OUT / "model/lightgbm/model.txt"))
    lgbm_manifest = {
        "source_files": [str(p.relative_to(ROOT)) for p in train_paths],
        "rows_before": rows_before, "rows_after": len(us),
        "positives": int(us.label.sum()), "negatives": int((us.label == 0).sum()),
        "unique_s1": int(us.s1.nunique()), "countries": us.country.value_counts().to_dict(),
        "feature_count": len(features), "features": features, "config": cfg,
        "zero_india_owned_rows": int(us.country.eq("India").sum()) == 0,
    }
    atomic_json(OUT / "training/us_only_lgbm_manifest.json", lgbm_manifest)
    del train, us; gc.collect()

    xtrain0 = pd.read_parquet(E004 / "gpu_bundle/train_pairs.parquet")
    xtrain = xtrain0[xtrain0.country.eq("US")].copy().reset_index(drop=True)
    atomic_parquet(xtrain, OUT / "training/us_only_xlmr_pairs.parquet")
    xmanifest = {
        "source": str((E004 / "gpu_bundle/train_pairs.parquet").relative_to(ROOT)),
        "rows_before": len(xtrain0), "rows_after": len(xtrain),
        "positives": int(xtrain.label.sum()), "negatives": int((xtrain.label == 0).sum()),
        "unique_s1": int(xtrain.source1_entity_id.nunique()),
        "countries": xtrain.country.value_counts().to_dict(),
        "zero_india_owned_rows": int(xtrain.country.eq("India").sum()) == 0,
    }
    atomic_json(OUT / "training/us_only_xlmr_manifest.json", xmanifest)
    del xtrain0, xtrain; gc.collect()

    # Materialize the historical calibration candidate pool with US-only LGBM scores/ranks.
    cal_bundle = pd.read_parquet(E004 / "gpu_bundle/calibration_pairs.parquet")
    cal_bundle = cal_bundle[cal_bundle.country.eq("US")].copy()
    raw = pd.concat([pd.read_parquet(p) for p in sorted((E001 / "cache/features").glob("calibration_US_*.parquet"))], ignore_index=True)
    raw["candidate_source"] = np.where(raw.source.eq(2), "S2", "S3")
    uk = ["s1", "target_id", "candidate_source"]
    if raw.duplicated(uk).any() or cal_bundle.duplicated(KEY).any(): raise RuntimeError("calibration pair keys not unique")
    score_map = raw.set_index(uk)[features]
    ix = pd.MultiIndex.from_frame(cal_bundle[["source1_entity_id", "candidate_entity_id", "candidate_source"]])
    score_features = score_map.reindex(ix)
    if score_features.isna().any().any(): raise RuntimeError("US calibration feature join incomplete")
    us_model = lgb.Booster(model_file=str(OUT / "model/lightgbm/model.txt"))
    cal_bundle["lightgbm_score"] = us_model.predict(score_features, num_threads=2).astype(np.float32)
    cal_bundle.sort_values(["source1_entity_id", "lightgbm_score", "candidate_entity_id"],
                           ascending=[True, False, True], kind="mergesort", inplace=True)
    cal_bundle["lightgbm_rank_within_s1"] = (cal_bundle.groupby("source1_entity_id", sort=False).cumcount() + 1).astype(np.int16)
    atomic_parquet(cal_bundle, OUT / "cache/calibration/us_only_candidates.parquet")
    cal_meta = hist_cal[hist_cal.country.eq("US")].copy()
    cal_meta.to_parquet(OUT / "cache/calibration/us_metadata.parquet", index=False, compression="zstd")
    atomic_json(OUT / "data_integrity.json", {
        "india_validation_s1": len(ids), "historical_calibration_s1": len(hist_cal),
        "us_calibration_s1": len(cal_meta), "overlap_validation": checks,
        "lgbm": lgbm_manifest, "xlmr": xmanifest,
        "validation_ids_sha256": sha256(OUT / "validation_ids.tsv"),
        "seconds": time.time() - t0,
    })
    print(json.dumps(json.loads((OUT / "data_integrity.json").read_text()), indent=2))


class FrozenIndex:
    def __init__(self, matrix):
        self.a = matrix.tocsr(); c = self.a.tocsc()
        self.c_indptr=run_r006a.i32(c.indptr); self.c_indices=run_r006a.i32(c.indices); self.c_data=run_r006a.f32(c.data)
        self.a_indptr=run_r006a.i32(self.a.indptr); self.a_indices=run_r006a.i32(self.a.indices); self.a_data=run_r006a.f32(self.a.data)
        self.fn, _ = run_r006b.load_functions()

    def query(self, xq):
        q=xq.tocsr(); qip=run_r006a.i32(q.indptr); qix=run_r006a.i32(q.indices); qv=run_r006a.f32(q.data)
        nq=q.shape[0]; oi=np.full((nq,10),-1,np.int32); ov=np.zeros((nq,10),np.float32)
        visits=np.zeros(nq,np.int64); full=np.zeros(nq,np.int64); counts=np.zeros(nq,np.int32); fallback=np.zeros(nq,np.int32); timing=np.zeros(5,np.float64)
        ip=ctypes.POINTER(ctypes.c_int32); fp=ctypes.POINTER(ctypes.c_float)
        rc=self.fn(qip.ctypes.data_as(ip),qix.ctypes.data_as(ip),qv.ctypes.data_as(fp),nq,
            self.c_indptr.ctypes.data_as(ip),self.c_indices.ctypes.data_as(ip),self.c_data.ctypes.data_as(fp),
            self.a_indptr.ctypes.data_as(ip),self.a_indices.ctypes.data_as(ip),self.a_data.ctypes.data_as(fp),self.a.shape[0],
            20000,800,2,oi.ctypes.data_as(ip),ov.ctypes.data_as(fp),visits.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            full.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),counts.ctypes.data_as(ip),fallback.ctypes.data_as(ip),
            timing.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if rc: raise RuntimeError(f"R006B kernel rc={rc}")
        return oi,ov


def build_index() -> None:
    t0=time.time(); init_dirs(); ids=pd.read_csv(OUT/"validation_ids.tsv",sep="\t")
    con=duckdb.connect(str(DB),read_only=True); con.register("ids",ids[["source1_entity_id"]])
    q=con.execute("""select s.entity_id,s.business_name,coalesce(s.business_address,'')business_address,s.nname,coalesce(s.naddr,'')naddr,s.country
      from train_s1 s join ids i on i.source1_entity_id=s.entity_id order by s.entity_id""").fetchdf()
    atomic_parquet(q,OUT/"cache/queries.parquet")
    for source in [2,3]:
        base=OUT/f"cache/index/s{source}"; base.mkdir(parents=True,exist_ok=True)
        aux=con.execute(f"""select a.entity_id,a.business_name,coalesce(a.business_address,'')business_address,a.nname,coalesce(a.naddr,'')naddr,a.country
          from train_s{source} a join read_parquet(?) f on f.entity_id=a.entity_id and f.source={source}
          where f.fold=4 and a.country='India' order by a.entity_id""",[str(ASSIGNMENT)]).fetchdf()
        nf=aux.nname.value_counts(dropna=False); af=aux.naddr.value_counts(dropna=False)
        aux["candidate_normalized_name_fanout"]=aux.nname.map(nf).clip(0,65535).astype(np.uint16)
        aux["candidate_normalized_address_fanout"]=aux.naddr.map(af).clip(0,65535).astype(np.uint16)
        atomic_parquet(aux,base/"aux.parquet")
        for channel,col in [("name","business_name"),("address","business_address")]:
            counts=e001.vec_for(channel).transform(aux[col].fillna("")); tf=TfidfTransformer()
            xa=tf.fit_transform(counts).astype(np.float32).tocsr(); sparse.save_npz(base/f"{channel}.npz",xa,compressed=False)
            np.save(base/f"{channel}_idf.npy",tf.idf_.astype(np.float32))
    con.close(); atomic_json(OUT/"runtime/index.json",{"seconds":time.time()-t0,"queries":len(q),"threads":2})


def qmatrix(q,channel,idf):
    tf=TfidfTransformer(); tf.idf_=idf; col="business_name" if channel=="name" else "business_address"
    return tf.transform(e001.vec_for(channel).transform(q[col].fillna(""))).astype(np.float32).tocsr()


def retrieve(worker:int,workers:int) -> None:
    t0=time.time(); qfull=pd.read_parquet(OUT/"cache/queries.parquet"); task=0; rows=0
    for source in [2,3]:
        base=OUT/f"cache/index/s{source}"; aux=pd.read_parquet(base/"aux.parquet",columns=["entity_id"])
        indexes={c:FrozenIndex(sparse.load_npz(base/f"{c}.npz")) for c in ["name","address"]}
        idfs={c:np.load(base/f"{c}_idf.npy") for c in ["name","address"]}
        for part,start in enumerate(range(0,len(qfull),SHARD)):
            mine=task%workers==worker; task+=1
            if not mine: continue
            op=OUT/f"cache/blocker/s{source}/part-{part:05d}.parquet"
            if op.exists(): rows+=len(pd.read_parquet(op,columns=["s1"])); continue
            q=qfull.iloc[start:start+SHARD].reset_index(drop=True); frames=[]; mats={}
            for channel in ["name","address"]:
                xq=qmatrix(q,channel,idfs[channel]); mats[channel]=xq; oi,ov=indexes[channel].query(xq)
                rr,kk=np.where(oi>=0); frames.append(pd.DataFrame({"qpos":rr.astype(np.int32),"apos":oi[rr,kk].astype(np.int32),
                    f"{channel}_rank":(kk+1).astype(np.int16),f"{channel}_retrieval_score":ov[rr,kk].astype(np.float32)}))
            cand=frames[0].merge(frames[1],on=["qpos","apos"],how="outer")
            cand["s1"]=q.entity_id.to_numpy()[cand.qpos]; cand["target_id"]=aux.entity_id.to_numpy()[cand.apos]; cand["source"]=np.int8(source)
            for channel in ["name","address"]:
                cand[f"{channel}_char_tfidf"]=e001.row_cosines(mats[channel],indexes[channel].a,cand.qpos.to_numpy(),cand.apos.to_numpy())
            atomic_parquet(cand,op); rows+=len(cand); print("BLOCKER",source,part,len(cand),flush=True)
    atomic_json(OUT/f"runtime/retrieve_worker_{worker}.json",{"seconds":time.time()-t0,"rows":rows,"worker":worker,"workers":workers})


def feature_tasks():
    n=math.ceil(len(pd.read_parquet(OUT/"cache/queries.parquet",columns=["entity_id"]))/SHARD)
    for source in [2,3]:
        for part in range(n): yield source,part


def features(worker:int,workers:int) -> None:
    t0=time.time(); qfull=pd.read_parquet(OUT/"cache/queries.parquet"); ids=pd.read_csv(OUT/"validation_ids.tsv",sep="\t")
    con=duckdb.connect(str(DB),read_only=True); std=lgb.Booster(model_file=str(STD_MODEL)); us=lgb.Booster(model_file=str(OUT/"model/lightgbm/model.txt"))
    fs=json.loads(MODEL_CONFIG.read_text())["features"]; done=rows=0
    for ti,(source,part) in enumerate(feature_tasks()):
        if ti%workers!=worker: continue
        op=OUT/f"cache/features/s{source}/part-{part:05d}.parquet"
        if op.exists(): continue
        base=OUT/f"cache/index/s{source}"; aux=pd.read_parquet(base/"aux.parquet"); q=qfull.iloc[part*SHARD:(part+1)*SHARD].reset_index(drop=True)
        cand=pd.read_parquet(OUT/f"cache/blocker/s{source}/part-{part:05d}.parquet")
        nf=aux.nname.value_counts(dropna=False); af=aux.naddr.value_counts(dropna=False)
        cand["country"]="India"; cand["role"]="clean_fold4_remainder"
        cand["s1_normalized_name_fanout"]=q.nname.map(nf).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
        cand["s1_normalized_address_fanout"]=q.naddr.map(af).fillna(0).clip(0,65535).astype(np.uint16).to_numpy()[cand.qpos]
        cand["candidate_normalized_name_fanout"]=aux.candidate_normalized_name_fanout.to_numpy()[cand.apos]
        cand["candidate_normalized_address_fanout"]=aux.candidate_normalized_address_fanout.to_numpy()[cand.apos]
        qn=q.nname.to_numpy()[cand.qpos]; qa=q.naddr.to_numpy()[cand.qpos]; an=aux.nname.to_numpy()[cand.apos]; aa=aux.naddr.to_numpy()[cand.apos]
        cand["nname_exact"]=((qn!="")&(qn==an)).astype(np.int8); cand["naddr_exact"]=((qa!="")&(qa==aa)).astype(np.int8)
        qids=q[["entity_id"]].rename(columns={"entity_id":"s1"}); con.register("qids",qids)
        truth=con.execute("select e.s1,e.target_id from edges e join qids q using(s1) where e.target_source=?",[source]).fetchdf(); con.unregister("qids")
        tset=set(map(tuple,truth.itertuples(index=False,name=None))); cand["label"]=np.fromiter(((a,b) in tset for a,b in zip(cand.s1,cand.target_id)),np.int8,len(cand))
        feat=build_features(add_context(cand),q,aux,workers=2); x=feat[fs].to_numpy()
        if not np.isfinite(x).all(): raise RuntimeError("nonfinite feature")
        cand["standard_lgbm_score"]=std.predict(feat[fs],num_threads=2).astype(np.float32)
        cand["us_only_lgbm_score"]=us.predict(feat[fs],num_threads=2).astype(np.float32)
        keep=["s1","target_id","source","country","label","name_rank","name_retrieval_score","address_rank","address_retrieval_score",
              "standard_lgbm_score","us_only_lgbm_score"]
        atomic_parquet(cand[keep],op); done+=1; rows+=len(cand); print("FEATURE",worker,source,part,len(cand),flush=True)
    con.close(); atomic_json(OUT/f"runtime/features_worker_{worker}.json",{"seconds":time.time()-t0,"parts":done,"rows":rows,"worker":worker})


def make_shortlists() -> None:
    t0=time.time(); q=pd.read_parquet(OUT/"cache/queries.parquet").set_index("entity_id")
    aux={s:pd.read_parquet(OUT/f"cache/index/s{s}/aux.parquet").set_index("entity_id") for s in [2,3]}
    parts=[]
    for source,part in feature_tasks(): parts.append(pd.read_parquet(OUT/f"cache/features/s{source}/part-{part:05d}.parquet"))
    d=pd.concat(parts,ignore_index=True); del parts
    for prefix in ["standard","us_only"]:
        score=f"{prefix}_lgbm_score"; rank=f"{prefix}_lgbm_rank"
        order=d[["s1",score,"target_id"]].sort_values(["s1",score,"target_id"],ascending=[True,False,True],kind="mergesort").index
        ranks=pd.Series(index=order,data=(d.loc[order].groupby("s1",sort=False).cumcount()+1).to_numpy(dtype=np.int16))
        d[rank]=ranks.sort_index().to_numpy()
        d[f"{prefix}_shortlist"] = d[rank].le(5) | d[score].ge(.05)
    d["candidate_source"]=np.where(d.source.eq(2),"S2","S3")
    d["s1_business_name"]=d.s1.map(q.business_name); d["s1_business_address"]=d.s1.map(q.business_address).fillna("")
    d["candidate_business_name"]=""; d["candidate_business_address"]=""
    for s in [2,3]:
        take=d.source.eq(s); z=d.loc[take,"target_id"]
        d.loc[take,"candidate_business_name"]=z.map(aux[s].business_name).to_numpy(); d.loc[take,"candidate_business_address"]=z.map(aux[s].business_address).fillna("").to_numpy()
    d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True)
    union=d.standard_shortlist|d.us_only_shortlist|d.label.eq(1)
    cols=KEY+["country","label","s1_business_name","s1_business_address","candidate_business_name","candidate_business_address",
        "standard_lgbm_score","us_only_lgbm_score","standard_lgbm_rank","us_only_lgbm_rank","standard_shortlist","us_only_shortlist"]
    atomic_parquet(d.loc[union,cols].reset_index(drop=True),OUT/"predictions/neural_scoring_union.parquet")
    atomic_parquet(d[[*KEY,"label","standard_lgbm_score","us_only_lgbm_score","standard_lgbm_rank","us_only_lgbm_rank","standard_shortlist","us_only_shortlist"]],OUT/"predictions/blocker_predictions.parquet")
    ids=pd.read_csv(OUT/"validation_ids.tsv",sep="\t")
    blocker=grouped_f05(d,ids,d.label.eq(1).to_numpy())
    atomic_json(OUT/"runtime/shortlist.json",{"seconds":time.time()-t0,"candidate_pairs":len(d),"candidates_per_s1":len(d)/len(ids),"blocker_oracle":blocker["macro_f05"],"union_rows":int(union.sum()),"standard_shortlist_rows":int(d.standard_shortlist.sum()),"us_only_shortlist_rows":int(d.us_only_shortlist.sum())})
    print(json.dumps(json.loads((OUT/"runtime/shortlist.json").read_text()),indent=2))


def finalize() -> None:
    """Select epoch on US calibration only, compute metrics/shifts, and write report."""
    t0=time.time(); ids=pd.read_csv(OUT/"validation_ids.tsv",sep="\t")
    calmeta=pd.read_parquet(OUT/"cache/calibration/us_metadata.parquet")
    epoch_rows=[]
    for ep in [1,2]:
        p=OUT/f"predictions/us_calibration_epoch{ep}.parquet"; d=pd.read_parquet(p)
        rows,best=search_policy(d,calmeta,"E005A")
        fixed_scores=.6*d.neural_probability.to_numpy()+.4*d.lightgbm_score.to_numpy()
        fixed=grouped_f05(d,calmeta,mask_for(d,fixed_scores,{"decoder":"one_owner_margin","threshold":.51,"margin":.48}))
        epoch_rows.append({"epoch":ep,"selection_e005a_macro_f05":best["macro_f05"],"selection_policy":best,"fixed_e005b":fixed})
    selected=max(epoch_rows,key=lambda z:z["selection_e005a_macro_f05"])["epoch"]
    atomic_json(OUT/"training/us_calibration_checkpoint_selection.json",{"criterion":"historical E005A best grouped macro F0.5, US calibration only","epochs":epoch_rows,"selected_epoch":selected})

    pairs=pd.read_parquet(OUT/"predictions/neural_scoring_union.parquet")
    stdn=pd.read_parquet(OUT/"predictions/standard_neural.parquet")[[*KEY,"neural_probability"]].rename(columns={"neural_probability":"standard_xlmr_score"})
    usn=pd.read_parquet(OUT/f"predictions/us_only_neural_epoch{selected}.parquet")[[*KEY,"neural_probability"]].rename(columns={"neural_probability":"us_only_xlmr_score"})
    pairs=pairs.merge(stdn,on=KEY,validate="one_to_one").merge(usn,on=KEY,validate="one_to_one")
    metrics={}; predcols={}
    blocker=pd.read_parquet(OUT/"predictions/blocker_predictions.parquet")
    for name in ["standard","us_only"]:
        take=pairs[f"{name}_shortlist"].to_numpy(); x=pairs.loc[take].copy()
        x["fused_score"]=(.6*x[f"{name}_xlmr_score"]+.4*x[f"{name}_lgbm_score"]).astype(np.float32)
        mask=mask_for(x,x.fused_score.to_numpy(),{"decoder":"one_owner_margin","threshold":.51,"margin":.48})
        actual=grouped_f05(x,ids,mask); oracle=grouped_f05(x,ids,x.label.eq(1).to_numpy())
        metrics[name]={**actual,"shortlist_oracle":oracle["macro_f05"],"oracle_model_gap":oracle["macro_f05"]-actual["macro_f05"],
            "true_positive_shortlist_survival":int(x.label.sum())/int(ids.true_matches.sum()),
            "predicted_matches_per_s1":actual["predicted_edges"]/len(ids),"true_matches_per_s1":actual["true_edges"]/len(ids),"shortlist_pairs":len(x)}
        pred=x.copy(); pred["accepted"]=mask; atomic_parquet(pred,OUT/f"predictions/{name}_india_predictions.parquet")
    blocker_oracle=grouped_f05(blocker,ids,blocker.label.eq(1).to_numpy())["macro_f05"]
    for z in metrics.values(): z["blocker_oracle"]=blocker_oracle
    atomic_json(OUT/"results.json",{"selected_epoch":selected,"metrics":metrics})

    dist=[]; qs=[.10,.25,.50,.75,.90]
    common=pairs.copy()
    for name in ["standard","us_only"]:
        common[f"{name}_fused_score"]=(.6*common[f"{name}_xlmr_score"]+.4*common[f"{name}_lgbm_score"]).astype(np.float32)
        for pair_type,mask in [("true",common.label.eq(1)),("hard_false",common.label.eq(0))]:
            for score,col in [("LightGBM",f"{name}_lgbm_score"),("XLM-R",f"{name}_xlmr_score"),("fused",f"{name}_fused_score")]:
                vals=common.loc[mask,col].quantile(qs).to_numpy()
                dist.append({"model":name,"score":score,"pair_type":pair_type,**{f"p{int(q*100)}":float(v) for q,v in zip(qs,vals)},"pairs":int(mask.sum())})
    pd.DataFrame(dist).to_csv(OUT/"score_distributions.csv",index=False)
    atomic_parquet(common,OUT/"predictions/score_distribution_pairs.parquet")

    drop=metrics["standard"]["macro_f05"]-metrics["us_only"]["macro_f05"]
    sensitivity="LOW" if drop<.010 else ("MODERATE" if drop<=.030 else "HIGH")
    integrity=json.loads((OUT/"data_integrity.json").read_text()); rt=json.loads((OUT/"runtime/shortlist.json").read_text())
    gaps=[]
    for k in ["blocker_oracle","shortlist_oracle","macro_f05","precision","recall","oracle_model_gap"]:
        gaps.append((k,metrics["standard"][k],metrics["us_only"][k],metrics["us_only"][k]-metrics["standard"][k]))
    lines=["# O001 US-only to India country-held-out diagnostic","","## DATA_INTEGRITY","",
      f"- India validation S1: {len(ids):,}",f"- LGBM US-only: {integrity['lgbm']['rows_after']:,} rows; {integrity['lgbm']['positives']:,} positive; {integrity['lgbm']['negatives']:,} negative; {integrity['lgbm']['unique_s1']:,} unique S1",
      f"- XLM-R US-only: {integrity['xlmr']['rows_after']:,} rows; {integrity['xlmr']['positives']:,} positive; {integrity['xlmr']['negatives']:,} negative; {integrity['xlmr']['unique_s1']:,} unique S1",
      f"- Validation overlap with LGBM train / XLM-R train / historical calibration: 0 / 0 / 0",f"- Selected US-only checkpoint: epoch {selected}, using US-only calibration only","",
      "## BLOCKER","",f"- Candidate count: {rt['candidate_pairs']:,}",f"- Candidates/S1: {rt['candidates_per_s1']:.6f}",f"- Blocker oracle: {blocker_oracle:.6f}",""]
    for name,title in [("standard","STANDARD_SEEN_INDIA"),("us_only","US_ONLY_UNSEEN_INDIA")]:
        m=metrics[name]; lines += [f"## {title}",""]+[f"- {k}: {m[k]:.6f}" for k in ["macro_f05","precision","recall","singleton_accuracy","shortlist_oracle","predicted_matches_per_s1","true_matches_per_s1"]]+[""]
    lines += ["## GENERALIZATION_GAP","","| metric | standard | US-only | delta (US-only - standard) |","|---|---:|---:|---:|"]
    lines += [f"| {k} | {a:.6f} | {b:.6f} | {d:+.6f} |" for k,a,b,d in gaps]
    lines += ["","## SCORE_SHIFT","","See `score_distributions.csv` for p10/p25/p50/p75/p90 on the identical neural-scoring pair union.","",
      "## INTERPRETATION","",f"- Sensitivity: **{sensitivity}** (standard minus US-only F0.5 = {drop:.6f}).",
      "- This is a structural unseen-country proxy, not an estimate of France performance.",
      "- Candidate generation is fixed and identical; observed downstream differences arise from supervised matcher exposure.",
      f"- The drop {'can' if drop>=.010 else 'does not by itself'} plausibly explain a meaningful fraction of the 0.040 local/public gap.","",
      "## RUNTIME","","Runtime details are recorded under `runtime/` and the two training logs.","",
      "## DECISION_RELEVANCE","",f"Unseen-country supervised generalization {'can plausibly account for a substantial portion' if drop>=.020 else 'does not appear sufficient by itself to account for a substantial portion'} of the 0.040 gap.",""]
    (OUT/"REPORT.md").write_text("\n".join(lines))
    atomic_json(OUT/"runtime/finalize.json",{"seconds":time.time()-t0,"sensitivity":sensitivity,"drop":drop})
    print((OUT/"REPORT.md").read_text())


def hashes() -> None:
    paths=[OUT/"validation_ids.tsv",OUT/"training/us_only_lgbm_manifest.json",OUT/"training/us_only_xlmr_manifest.json",
      OUT/"model/lightgbm/model.txt",OUT/"model/xlmr/best_checkpoint/model.safetensors",OUT/"results.json",
      OUT/"predictions/blocker_predictions.parquet",OUT/"predictions/standard_india_predictions.parquet",
      OUT/"predictions/us_only_india_predictions.parquet",OUT/"predictions/score_distribution_pairs.parquet",
      OUT/"score_distributions.csv",OUT/"correctness_audit.json",OUT/"REPORT.md"]
    rows=[{"path":str(p.relative_to(OUT)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in paths]
    atomic_json(OUT/"hashes.json",rows); print(json.dumps(rows,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","index","retrieve","features","shortlist","finalize","hashes"])
    ap.add_argument("--worker",type=int,default=0); ap.add_argument("--workers",type=int,default=1); a=ap.parse_args()
    {"prepare":prepare,"index":build_index,"retrieve":lambda:retrieve(a.worker,a.workers),"features":lambda:features(a.worker,a.workers),
     "shortlist":make_shortlists,"finalize":finalize,"hashes":hashes}[a.stage]()


if __name__ == "__main__": main()
