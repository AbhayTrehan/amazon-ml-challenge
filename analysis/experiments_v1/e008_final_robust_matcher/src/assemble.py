#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import *

FEATURES=json.loads((E001/"model/model_config.json").read_text())["features"]
NON_DIGIT_TOKEN=re.compile(r"[^\W\d_]+|\d+|[^\w\s]",re.UNICODE)


def feature_files(role):return sorted((EXP/f"cache/features/{role}").glob("*/*/part-*.parquet"))


def text_for_pairs(pairs):
    x=pairs.reset_index(drop=True).copy();x["_rid"]=np.arange(len(x),dtype=np.int64)
    con=duckdb.connect(str(DB),read_only=True);con.register("pairs",x)
    out=con.execute("""select p.* exclude(_rid),q.business_name s1_business_name,coalesce(q.business_address,'')s1_business_address,
      q.nname s1_nname,coalesce(q.naddr,'')s1_naddr,
      a.business_name candidate_business_name,coalesce(a.business_address,'')candidate_business_address,
      a.nname candidate_nname,coalesce(a.naddr,'')candidate_naddr
      from pairs p join train_s1 q on q.entity_id=p.source1_entity_id
      join (select 'S2' candidate_source,* exclude(row_num) from train_s2 union all by name select 'S3' candidate_source,* exclude(row_num) from train_s3) a
      using(candidate_source) where a.entity_id=p.candidate_entity_id order by p._rid""").fetchdf();con.close()
    out["s1_numbers"]=out.s1_naddr.map(numeric_tokens);out["candidate_numbers"]=out.candidate_naddr.map(numeric_tokens)
    if len(out)!=len(x):raise RuntimeError("pair text join changed row count")
    return out


def export_historical_inputs():
    for fp in feature_files("train"):
        rel=fp.relative_to(EXP/"cache/features/train");op=EXP/"cache/neural/historical_inputs"/rel
        if op.exists():continue
        d=pd.read_parquet(fp,columns=["s1","target_id","source"]);d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True);d["candidate_source"]=np.where(d.source.eq(2),"S2","S3")
        atomic_parquet(text_for_pairs(d),op);print("TEXT",rel,len(d),flush=True)


def mining_table():
    rows=[]
    for fp in feature_files("train"):
        rel=fp.relative_to(EXP/"cache/features/train");npth=EXP/"cache/neural/historical"/rel
        d=pd.read_parquet(fp,columns=["s1","target_id","source","country","label","historical_lgbm","name_retrieval_score","address_retrieval_score","nname_exact","naddr_exact","fused_max_score"])
        n=pd.read_parquet(npth);d["historical_neural"]=n.historical_neural.to_numpy();d["shard"]=str(rel);d["row_in_shard"]=np.arange(len(d),dtype=np.int32);rows.append(d)
    d=pd.concat(rows,ignore_index=True);d["historical_fused"]=(.6*d.historical_neural+.4*d.historical_lgbm).astype(np.float32)
    d=d.sort_values(["s1","historical_lgbm","target_id"],ascending=[True,False,True],kind="mergesort");d["historical_lgbm_rank"]=(d.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    return d


def choose_negatives(d):
    neg=d[d.label.eq(0)].copy();chosen=[]
    top=neg.sort_values(["s1","historical_fused","target_id"],ascending=[True,False,True],kind="mergesort");top=top[top.groupby("s1",sort=False).cumcount()<4].copy();top["category"]="model_top4";chosen.append(top)
    used=pd.MultiIndex.from_frame(top[["s1","target_id","source"]]);rem=neg[~pd.MultiIndex.from_frame(neg[["s1","target_id","source"]]).isin(used)].copy()
    rem["name_priority"]=rem.nname_exact*10+rem.name_retrieval_score
    name=rem.sort_values(["s1","name_priority","historical_fused","target_id"],ascending=[True,False,False,True],kind="mergesort");name=name[name.groupby("s1",sort=False).cumcount()<1].copy();name["category"]="name_confusable";chosen.append(name)
    used=used.append(pd.MultiIndex.from_frame(name[["s1","target_id","source"]]));rem=neg[~pd.MultiIndex.from_frame(neg[["s1","target_id","source"]]).isin(used)].copy()
    rem["address_priority"]=rem.naddr_exact*10+rem.address_retrieval_score
    addr=rem.sort_values(["s1","address_priority","historical_fused","target_id"],ascending=[True,False,False,True],kind="mergesort");addr=addr[addr.groupby("s1",sort=False).cumcount()<1].copy();addr["category"]="address_confusable";chosen.append(addr)
    out=pd.concat(chosen,ignore_index=True);out=out.drop_duplicates(["s1","target_id","source"])
    counts=out.groupby("s1").size();need=(6-counts).clip(lower=0);used=pd.MultiIndex.from_frame(out[["s1","target_id","source"]]);rem=neg[~pd.MultiIndex.from_frame(neg[["s1","target_id","source"]]).isin(used)].copy()
    rem=rem.sort_values(["s1","historical_lgbm","target_id"],ascending=[True,False,True],kind="mergesort");rem["_rank"]=rem.groupby("s1",sort=False).cumcount();rem["_need"]=rem.s1.map(need).fillna(6).astype(int);fill=rem[rem._rank<rem._need].copy();fill["category"]="lgbm_fill"
    out=pd.concat([out,fill],ignore_index=True).drop_duplicates(["s1","target_id","source"])
    return out


def hard_positives(d):
    p=d[d.label.eq(1)].copy();p["candidate_source"]=np.where(p.source.eq(2),"S2","S3")
    qf=p.groupby(["country","source"]).historical_fused.transform(lambda x:x.quantile(.25));ql=p.groupby(["country","source"]).fused_max_score.transform(lambda x:x.quantile(.25))
    absent=~(p.historical_lgbm_rank.le(5)|p.historical_lgbm.ge(.05))
    p["hard_bottom_fused"]=p.historical_fused.le(qf);p["hard_absent_shortlist"]=absent;p["hard_lexical_tail"]=p.fused_max_score.le(ql);p["hard"]=p[["hard_bottom_fused","hard_absent_shortlist","hard_lexical_tail"]].any(axis=1)
    return p


def augment_text(row):
    seed=stable_hash(f"{row['source1_entity_id']}|{row['candidate_entity_id']}|{row['candidate_source']}","e008-hard-positive-augmentation-20260925");rng=random.Random(seed)
    side=rng.choice(["s1","candidate"]);name=str(row[f"{side}_business_name"] or "");addr=str(row[f"{side}_business_address"] or "")
    ops=["case","punct","space","suffix","addr_delete"];op=ops[seed%len(ops)]
    if op=="case":name=name.swapcase()
    elif op=="punct":name=re.sub(r"[^\w\s\d]",lambda m:" " if (ord(m.group())+seed)%2 else "-",name)
    elif op=="space":name=re.sub(r"\s+","  ",name.strip())
    elif op=="suffix":
        toks=name.split();
        if len(toks)>1 and normalize_existing(toks[-1]) in LEGAL:name=" ".join(toks[:-1])
        else:name=name.swapcase()
    else:
        toks=addr.split();eligible=[i for i,t in enumerate(toks) if not any(c.isdigit() for c in t)];n=min(len(eligible),max(0,int(len(toks)*.15)))
        if n and len(toks)-n>0:
            drop=set(rng.sample(eligible,n));addr=" ".join(t for i,t in enumerate(toks) if i not in drop)
        else:name=name.swapcase()
    if not name.strip():name=str(row[f"{side}_business_name"] or "")
    if not addr.strip():addr=str(row[f"{side}_business_address"] or "")
    row[f"{side}_business_name"]=name;row[f"{side}_business_address"]=addr;row[f"{side}_nname"]=normalize_existing(name);row[f"{side}_naddr"]=normalize_existing(addr);row[f"{side}_numbers"]=numeric_tokens(addr);row["augmentation_operation"]=op;row["augmentation_side"]=side
    return row


def assemble():
    t0=time.time();d=mining_table();neg6=choose_negatives(d);hp=hard_positives(d);pos=hp.copy()
    # LGBM negatives: the six prescribed model/lexical hard rows, then strongest remaining by max hard signal to ten.
    neg=d[d.label.eq(0)].copy();keys6=pd.MultiIndex.from_frame(neg6[["s1","target_id","source"]]);neg["selected6"]=pd.MultiIndex.from_frame(neg[["s1","target_id","source"]]).isin(keys6)
    neg["lgbm_priority"]=np.maximum.reduce([neg.historical_fused.to_numpy(),neg.name_retrieval_score.to_numpy(),neg.address_retrieval_score.to_numpy()])
    neg=neg.sort_values(["s1","selected6","lgbm_priority","target_id"],ascending=[True,False,False,True],kind="mergesort");neg10=neg[neg.groupby("s1",sort=False).cumcount()<10]
    keys=pd.concat([pos[["s1","target_id","source"]],neg10[["s1","target_id","source"]]],ignore_index=True).drop_duplicates();keys.to_parquet(EXP/"data/lgbm_training_keys.parquet",index=False)
    con=duckdb.connect();con.register("keys",keys);glob=str(EXP/"cache/features/train/*/s*/part-*.parquet")
    ltrain=con.execute(f"select f.* from read_parquet('{glob}',union_by_name=true) f join keys k using(s1,target_id,source)").fetchdf();con.close()
    model_cfg=json.loads((E001/"model/model_config.json").read_text())["config"];model=lgb.LGBMClassifier(**model_cfg);lt=time.time();model.fit(ltrain[FEATURES],ltrain.label);lsec=time.time()-lt
    joblib.dump(model,EXP/"model/lightgbm/model.joblib");model.booster_.save_model(str(EXP/"model/lightgbm/model.txt"))
    # Build XLM-R bundle: every real positive, selected six negatives, plus one deterministic copy per hard positive.
    base=pd.concat([pos[["s1","target_id","source","label"]],neg6[["s1","target_id","source","label"]]],ignore_index=True);base.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True);base["candidate_source"]=np.where(base.source.eq(2),"S2","S3");base["augmented"]=False
    txt=text_for_pairs(base);hardkeys=set(map(tuple,hp[hp.hard][["s1","target_id","source"]].itertuples(index=False,name=None)));aug=txt[[tuple(x) in hardkeys for x in txt[["source1_entity_id","candidate_entity_id","source"]].itertuples(index=False,name=None)]].copy();aug["augmented"]=True
    aug=pd.DataFrame([augment_text(r.copy()) for r in aug.to_dict("records")]);bundle=pd.concat([txt,aug],ignore_index=True);bundle["relative_weight"]=np.where(bundle.augmented,1.5,1.0);bundle["row_weight"]=bundle.relative_weight/bundle.groupby("source1_entity_id").relative_weight.transform("sum")
    atomic_parquet(bundle,EXP/"data/xlmr_training.parquet");atomic_parquet(aug.head(100),EXP/"data/augmentation_audit_100.parquet")
    cat_sets={c:set(map(tuple,g[["s1","target_id","source"]].itertuples(index=False,name=None))) for c,g in neg6.groupby("category")};overlap={f"{a}__{b}":len(cat_sets[a]&cat_sets[b]) for a in cat_sets for b in cat_sets if a<b}
    report={"rows":len(bundle),"real_positives":len(pos),"augmented_positives":len(aug),"negatives":len(neg6),"negative_mean_per_s1":len(neg6)/200000,
      "negative_categories":neg6.category.value_counts().to_dict(),"category_overlap":overlap,"negative_fused_percentiles":neg6.historical_fused.quantile([.01,.1,.5,.9,.99]).to_dict(),
      "hard_positive_reasons":hp[["hard_bottom_fused","hard_absent_shortlist","hard_lexical_tail"]].sum().to_dict(),"lgbm_rows":len(ltrain),"lgbm_positives":int(ltrain.label.sum()),"lgbm_negatives":int((ltrain.label==0).sum()),"lgbm_seconds":lsec,"assemble_seconds":time.time()-t0,
      "lightgbm_sha256":sha256(EXP/"model/lightgbm/model.txt")}
    atomic_json(EXP/"data/training_report.json",report);print(json.dumps(report,indent=2),flush=True)


def score_fold4():
    model=joblib.load(EXP/"model/lightgbm/model.joblib");rows=[];t0=time.time()
    keep=["s1","target_id","source","country","label","historical_lgbm","name_retrieval_score","name_rank","address_retrieval_score","address_rank","nname_exact","naddr_exact","numeric_token_jaccard","numeric_exact_overlap_count","obvious_numeric_conflict","s1_normalized_name_fanout","candidate_address_missing"]
    for fp in feature_files("fold4"):
        d=pd.read_parquet(fp);d["lightgbm_score"]=model.predict_proba(d[FEATURES])[:,1].astype(np.float32);rows.append(d[keep+["lightgbm_score"]])
    d=pd.concat(rows,ignore_index=True);d=d.sort_values(["s1","lightgbm_score","target_id"],ascending=[True,False,True],kind="mergesort");d["lightgbm_rank_within_s1"]=(d.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    d=d[d.lightgbm_rank_within_s1.le(5)|d.lightgbm_score.ge(.05)].copy();d.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True);d["candidate_source"]=np.where(d.source.eq(2),"S2","S3")
    ids=pd.read_parquet(EXP/"data/fold4_s1.parquet",columns=["source1_entity_id","is_calibration"]);calset=set(ids[ids.is_calibration].source1_entity_id)
    for role,take in [("calibration",d.source1_entity_id.isin(calset)),("validation",~d.source1_entity_id.isin(calset))]:
        out=text_for_pairs(d.loc[take].reset_index(drop=True));atomic_parquet(out,EXP/f"cache/shortlist/{role}.parquet");print("SHORTLIST",role,len(out),flush=True)
    # P002 control on the identical blocker uses its own historical LightGBM shortlist.
    b=pd.concat(rows,ignore_index=True);b=b.sort_values(["s1","historical_lgbm","target_id"],ascending=[True,False,True],kind="mergesort");b["lightgbm_rank_within_s1"]=(b.groupby("s1",sort=False).cumcount()+1).astype(np.int16)
    b=b[b.lightgbm_rank_within_s1.le(5)|b.historical_lgbm.ge(.05)].copy();b.rename(columns={"s1":"source1_entity_id","target_id":"candidate_entity_id"},inplace=True);b["candidate_source"]=np.where(b.source.eq(2),"S2","S3")
    atomic_parquet(text_for_pairs(b[~b.source1_entity_id.isin(calset)].reset_index(drop=True)),EXP/"cache/shortlist/p002_validation.parquet")
    atomic_json(EXP/"runtime/fold4_lgbm.json",{"full_candidate_rows":sum(len(x) for x in rows),"shortlist_rows":len(d),"seconds":time.time()-t0})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("stage",choices=["export-historical","assemble","score-fold4"]);a=ap.parse_args();{"export-historical":export_historical_inputs,"assemble":assemble,"score-fold4":score_fold4}[a.stage]()
if __name__=="__main__":main()
