#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from common import *

PAIR_FEATURES=["xlmr_logit","lgbm_logit","source3","name_retrieval_score","address_retrieval_score","name_rank","address_rank","nname_exact","naddr_exact","numeric_token_jaccard","numeric_exact_overlap_count","obvious_numeric_conflict"]
OWNER_FEATURES=["best_pair_probability","second_pair_probability","logit_gap","probability_ratio","source3","best_lexical_retrieval_score"]


def pair_matrix(d):
    x=pd.DataFrame({"xlmr_logit":logit(d.xlmr_probability),"lgbm_logit":logit(d.lightgbm_score),"source3":d.candidate_source.eq("S3").astype(np.int8),
      "name_retrieval_score":d.name_retrieval_score,"address_retrieval_score":d.address_retrieval_score,"name_rank":d.name_rank,"address_rank":d.address_rank,
      "nname_exact":d.nname_exact,"naddr_exact":d.naddr_exact,"numeric_token_jaccard":d.numeric_token_jaccard,"numeric_exact_overlap_count":d.numeric_exact_overlap_count,"obvious_numeric_conflict":d.obvious_numeric_conflict})
    return x[PAIR_FEATURES]


def owner_examples(d):
    z=d.sort_values(["candidate_entity_id","pair_probability","source1_entity_id"],ascending=[True,False,True],kind="mergesort").copy();z["owner_rank"]=z.groupby("candidate_entity_id",sort=False).cumcount();counts=z.groupby("candidate_entity_id").size();w=z[z.owner_rank.eq(0)].copy();second=z[z.owner_rank.eq(1)].set_index("candidate_entity_id").pair_probability
    w["best_pair_probability"]=w.pair_probability;w["second_pair_probability"]=w.candidate_entity_id.map(second).fillna(0.0);w["logit_gap"]=logit(w.best_pair_probability)-logit(w.second_pair_probability);w["probability_ratio"]=w.best_pair_probability/np.clip(w.second_pair_probability,1e-6,None);w["source3"]=w.candidate_source.eq("S3").astype(np.int8);w["best_lexical_retrieval_score"]=w[["name_retrieval_score","address_retrieval_score"]].max(axis=1);w["contested"]=w.candidate_entity_id.map(counts).gt(1);return w


def apply_owner(d,owner):
    w=owner_examples(d);w["owner_probability"]=1.0;take=w.contested
    if take.any():w.loc[take,"owner_probability"]=owner.predict_proba(w.loc[take,OWNER_FEATURES])[:,1]
    w["effective_probability"]=w.pair_probability*w.owner_probability;return w


def old_decoder(d):
    z=d.sort_values(["candidate_entity_id","pair_probability","source1_entity_id"],ascending=[True,False,True],kind="mergesort").copy();z["rank"]=z.groupby("candidate_entity_id",sort=False).cumcount();sec=z[z["rank"].eq(1)].set_index("candidate_entity_id").pair_probability;w=z[z["rank"].eq(0)].copy();w["margin"]=w.pair_probability-w.candidate_entity_id.map(sec).fillna(0);return w,(w.pair_probability.ge(.51)&w.margin.ge(.48)).to_numpy()


def calibration_ids():
    ids=pd.read_parquet(EXP/"data/fold4_s1.parquet").query("is_calibration")[["source1_entity_id","country","fold"]]
    import duckdb
    con=duckdb.connect(str(DB),read_only=True);con.register("ids",ids)
    out=con.execute("select i.*,count(e.target_id)::integer true_matches from ids i left join edges e on e.s1=i.source1_entity_id group by all").fetchdf();con.close();return out


def run():
    base=pd.read_parquet(EXP/"cache/shortlist/calibration.parquet");selected=calibration_ids();rows=[];models={}
    for epoch in [1,2,3]:
        prob=pd.read_parquet(EXP/f"cache/neural/calibration_epoch{epoch}.parquet");d=base.copy();d["xlmr_probability"]=prob.xlmr_probability.to_numpy()
        stack=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000,random_state=SEED);stack.fit(pair_matrix(d),d.label);d["pair_probability"]=stack.predict_proba(pair_matrix(d))[:,1]
        owners=owner_examples(d);contest=owners[owners.contested];owner=LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=1000,random_state=SEED);owner.fit(contest[OWNER_FEATURES],contest.label)
        win=apply_owner(d,owner);mask=expected_utility_decode(win);met=grouped_metrics(win,selected,mask);ow,om=old_decoder(d);old=grouped_metrics(ow,selected,om)
        diag={"epoch":epoch,**met,"old_decoder_macro_f05":old["macro_f05"],"old_decoder_precision":old["precision"],
          "pair_brier":brier_score_loss(d.label,d.pair_probability),"pair_nll":log_loss(d.label,d.pair_probability,labels=[0,1]),
          "owner_rows":len(contest),"owner_accuracy":float(np.mean((owner.predict_proba(contest[OWNER_FEATURES])[:,1]>=.5)==contest.label.to_numpy()))}
        rows.append(diag);models[epoch]=(stack,owner,d,win,mask)
    best=max(rows,key=lambda x:(x["macro_f05"],x["precision"],-x["epoch"]));epoch=best["epoch"];stack,owner,d,win,mask=models[epoch]
    joblib.dump(stack,EXP/"calibration/pair_stacker.joblib");joblib.dump(owner,EXP/"calibration/owner_calibrator.joblib");(EXP/"model/xlmr/selected_epoch.txt").write_text(str(epoch)+"\n")
    coeff={"pair":{"features":PAIR_FEATURES,"intercept":stack.intercept_.tolist(),"coefficients":stack.coef_[0].tolist()},"owner":{"features":OWNER_FEATURES,"intercept":owner.intercept_.tolist(),"coefficients":owner.coef_[0].tolist()}}
    atomic_json(EXP/"calibration/coefficients.json",coeff);pd.DataFrame(rows).to_csv(EXP/"calibration/checkpoint_selection.csv",index=False)
    atomic_json(EXP/"calibration/summary.json",{"selected_epoch":epoch,"selection":"macro_f05, then precision, then earlier epoch","epochs":rows})
    freeze={"status":"FROZEN_BEFORE_VALIDATION","training_ids_sha256":sha256(EXP/"data/training_ids.txt"),"lightgbm_sha256":sha256(EXP/"model/lightgbm/model.txt"),
      "xlmr_checkpoint":f"model/xlmr/epoch-{epoch}","xlmr_sha256":sha256(EXP/f"model/xlmr/epoch-{epoch}/model.safetensors"),"normalization":"audit DuckDB lower + Unicode letter/number punctuation-to-space","serialization":"e008_fieldaware_v1","max_length":160,"frozen_layers":[0,1,2,3,4,5],
      "pair_stacker_sha256":sha256(EXP/"calibration/pair_stacker.joblib"),"owner_calibrator_sha256":sha256(EXP/"calibration/owner_calibrator.joblib"),"decoder":"max over k=0..n; U0=product(1-p); Uk=1.25*sum_topk(p)/(k+0.25*sum_all(p)); ties smaller k"}
    atomic_json(EXP/"freeze/frozen_config.json",freeze);print(json.dumps({"selected":best,"freeze":freeze},indent=2))


if __name__=="__main__":run()
