#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import *


def _head_tail(ids, n):
    if len(ids)<=n:return ids
    h=(n+1)//2;return ids[:h]+ids[-(n-h):]


class FieldAwareEncoder:
    """Deterministic 160-token pair encoding with protected names/numbers and address head+tail."""
    def __init__(self, tokenizer, max_length=160): self.tok=tokenizer;self.max_length=max_length
    def enc(self,text): return self.tok.encode("" if pd.isna(text) else str(text),add_special_tokens=False)
    def entity(self,row,prefix,source=None):
        fields=[("name",row[f"{prefix}_business_name"],18,"head"),("normalized name",row[f"{prefix}_nname"],18,"head"),
                ("address",row[f"{prefix}_business_address"],14,"tail"),("normalized address",row[f"{prefix}_naddr"],12,"tail"),
                ("numbers",row[f"{prefix}_numbers"],10,"head")]
        if source is not None: fields.insert(0,("source",source,4,"head"))
        pieces=[];used=0;cap=78
        raw=[]
        for label,value,budget,mode in fields:
            lab=self.enc(label+": ");val=self.enc(value);take=max(0,budget-len(lab));val=_head_tail(val,take) if mode=="tail" else val[:take]
            z=lab+val;raw.append((z,mode));used+=len(z)
        spare=max(0,cap-used)
        # Deterministically spend spare capacity on addresses, preserving both ends.
        for j in range(len(raw)):
            if spare<=0:break
            label,value,budget,mode=fields[j]
            if mode!="tail":continue
            lab=self.enc(label+": ");full=self.enc(value);cur=max(0,budget-len(lab));extra=min(spare,max(0,len(full)-cur));raw[j]=(lab+_head_tail(full,cur+extra),mode);spare-=extra
        for z,_ in raw: pieces.extend(z)
        return pieces[:cap]
    def encode_row(self,row):
        a=self.entity(row,"s1");b=self.entity(row,"candidate",str(row["candidate_source"]))
        ids=[self.tok.bos_token_id]+a+[self.tok.eos_token_id,self.tok.eos_token_id]+b+[self.tok.eos_token_id]
        if len(ids)>self.max_length: raise RuntimeError(f"field-aware encoding exceeded {self.max_length}")
        return ids


class EncodedDataset(Dataset):
    def __init__(self,ids,labels=None,weights=None):self.ids=ids;self.labels=labels;self.weights=weights
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return i


class Collator:
    def __init__(self,ds,tok):self.ds=ds;self.tok=tok
    def __call__(self,rows):
        x=self.tok.pad([{"input_ids":self.ds.ids[i],"attention_mask":[1]*len(self.ds.ids[i])} for i in rows],padding=True,pad_to_multiple_of=8,return_tensors="pt")
        if self.ds.labels is not None:x["labels"]=torch.tensor(self.ds.labels[rows],dtype=torch.float32)
        if self.ds.weights is not None:x["row_weights"]=torch.tensor(self.ds.weights[rows],dtype=torch.float32)
        return x


def encode_frame(frame,tok,cache=None):
    if cache and cache.exists():
        z=torch.load(cache,map_location="cpu",weights_only=False)
        if z.get("serialization_version")=="e008_fieldaware_v1" and z.get("rows")==len(frame):return z["ids"],z["truncated"]
    ids=[];trunc=0;chunk_size=16384
    specs_a=[("name","s1_business_name",18,"head"),("normalized name","s1_nname",18,"head"),("address","s1_business_address",14,"tail"),("normalized address","s1_naddr",12,"tail"),("numbers","s1_numbers",10,"head")]
    specs_b=[("source","candidate_source",4,"head"),("name","candidate_business_name",18,"head"),("normalized name","candidate_nname",18,"head"),("address","candidate_business_address",14,"tail"),("normalized address","candidate_naddr",12,"tail"),("numbers","candidate_numbers",10,"head")]
    labels={label:tok.encode(label+": ",add_special_tokens=False) for label,_,_,_ in specs_a+specs_b}

    def assemble(fields,specs,i):
        raw=[];used=0;cap=78
        for label,col,budget,mode in specs:
            lab=labels[label];full=fields[col][i];take=max(0,budget-len(lab));val=_head_tail(full,take) if mode=="tail" else full[:take]
            z=lab+val;raw.append((z,mode,col,budget));used+=len(z)
        spare=max(0,cap-used)
        for j,(z,mode,col,budget) in enumerate(raw):
            if spare<=0:break
            if mode!="tail":continue
            lab=labels[specs[j][0]];full=fields[col][i];cur=max(0,budget-len(lab));extra=min(spare,max(0,len(full)-cur));raw[j]=(lab+_head_tail(full,cur+extra),mode,col,budget);spare-=extra
        out=[]
        for z,_,_,_ in raw:out.extend(z)
        return out[:cap]

    for start in range(0,len(frame),chunk_size):
        x=frame.iloc[start:start+chunk_size]
        fields={}
        for _,col,_,_ in specs_a+specs_b:
            if col not in fields:
                values=x[col].fillna("").astype(str).tolist();fields[col]=tok(values,add_special_tokens=False,padding=False,truncation=False)["input_ids"]
        # Actual untruncated pair lengths for the complete field-aware serialization.
        atext=("name: "+x.s1_business_name.fillna("").astype(str)+" normalized name: "+x.s1_nname.fillna("").astype(str)+" address: "+x.s1_business_address.fillna("").astype(str)+" normalized address: "+x.s1_naddr.fillna("").astype(str)+" numbers: "+x.s1_numbers.fillna("").astype(str)).tolist()
        btext=("source: "+x.candidate_source.fillna("").astype(str)+" name: "+x.candidate_business_name.fillna("").astype(str)+" normalized name: "+x.candidate_nname.fillna("").astype(str)+" address: "+x.candidate_business_address.fillna("").astype(str)+" normalized address: "+x.candidate_naddr.fillna("").astype(str)+" numbers: "+x.candidate_numbers.fillna("").astype(str)).tolist()
        lengths=tok(atext,btext,add_special_tokens=True,padding=False,truncation=False,return_length=True)["length"];trunc+=sum(n>160 for n in lengths)
        for i in range(len(x)):
            a=assemble(fields,specs_a,i);b=assemble(fields,specs_b,i);pair=[tok.bos_token_id]+a+[tok.eos_token_id,tok.eos_token_id]+b+[tok.eos_token_id]
            if len(pair)>160:raise RuntimeError("field-aware encoding exceeded 160")
            ids.append(pair)
    if cache: cache.parent.mkdir(parents=True,exist_ok=True);torch.save({"serialization_version":"e008_fieldaware_v1","rows":len(frame),"ids":ids,"truncated":trunc},cache)
    return ids,trunc


def old_serialize(name,address):
    addr="[missing]" if pd.isna(address) or str(address).strip()=="" else str(address);name="" if pd.isna(name) else str(name)
    return f"name: {name}\naddress: {addr}"


def infer_old(input_path,output_path,batch=512):
    d=pd.read_parquet(input_path);tok=AutoTokenizer.from_pretrained(E005/"model/best_checkpoint",local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(E005/"model/best_checkpoint",local_files_only=True).cuda().eval();probs=[]
    for st in range(0,len(d),batch):
        x=d.iloc[st:st+batch];a=[old_serialize(n,z) for n,z in zip(x.s1_business_name,x.s1_business_address)];b=[old_serialize(n,z) for n,z in zip(x.candidate_business_name,x.candidate_business_address)]
        bt=tok(a,b,truncation=True,max_length=104,padding=True,pad_to_multiple_of=8,return_tensors="pt");bt={k:v.cuda() for k,v in bt.items()}
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):probs.append(torch.softmax(model(**bt).logits.float(),1)[:,1].cpu().numpy())
    out=pd.DataFrame({"historical_neural":np.concatenate(probs).astype(np.float32)});atomic_parquet(out,Path(output_path))


def infer_old_all(batch=512):
    root=EXP/"cache/neural/historical_inputs";files=sorted(root.glob("*/*/part-*.parquet"));tok=AutoTokenizer.from_pretrained(E005/"model/best_checkpoint",local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(E005/"model/best_checkpoint",local_files_only=True).cuda().eval()
    for ip in files:
        rel=ip.relative_to(root);op=EXP/"cache/neural/historical"/rel
        if op.exists():continue
        d=pd.read_parquet(ip);probs=[];t0=time.time()
        for st in range(0,len(d),batch):
            x=d.iloc[st:st+batch];a=[old_serialize(n,z) for n,z in zip(x.s1_business_name,x.s1_business_address)];b=[old_serialize(n,z) for n,z in zip(x.candidate_business_name,x.candidate_business_address)]
            bt=tok(a,b,truncation=True,max_length=104,padding=True,pad_to_multiple_of=8,return_tensors="pt");bt={k:v.cuda() for k,v in bt.items()}
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):probs.append(torch.softmax(model(**bt).logits.float(),1)[:,1].cpu().numpy())
        atomic_parquet(pd.DataFrame({"historical_neural":np.concatenate(probs).astype(np.float32)}),op)
        print("HISTORICAL_NEURAL",rel,len(d),time.time()-t0,flush=True)


def freeze_lower(model):
    base=model.roberta
    for p in base.embeddings.parameters():p.requires_grad=False
    for layer in base.encoder.layer[:6]:
        for p in layer.parameters():p.requires_grad=False


def model_and_tokenizer(checkpoint=None):
    src=checkpoint or MODEL_ID;kwargs={"local_files_only":True} if checkpoint else {"revision":MODEL_REVISION}
    tok=AutoTokenizer.from_pretrained(src,**kwargs);model=AutoModelForSequenceClassification.from_pretrained(src,num_labels=1,problem_type="multi_label_classification",**kwargs)
    freeze_lower(model);return model,tok


def optimizer_for(model):
    decay=[];nodec=[]
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        (nodec if n.endswith("bias") or "LayerNorm.weight" in n else decay).append(p)
    return AdamW([{"params":decay,"weight_decay":.01},{"params":nodec,"weight_decay":0}],lr=2e-5,fused=True)


def save_model(model,tok,path,state):
    path.mkdir(parents=True,exist_ok=True);model.save_pretrained(path,safe_serialization=True);tok.save_pretrained(path);torch.save(state,path/"training_state.pt")


def train():
    seed_everything();d=pd.read_parquet(EXP/"data/xlmr_training.parquet");model,tok=model_and_tokenizer();ids,trunc=encode_frame(d,tok,EXP/"runtime/train_tokens.pt")
    labels=d.label.to_numpy(np.float32);weights=d.row_weight.to_numpy(np.float32);ds=EncodedDataset(ids,labels,weights)
    # Short OOM-only engineering preflight, descending candidates.
    physical=None;pre=[]
    for bs in [512,384,256,192,128]:
        try:
            model.cuda().train();bt=Collator(ds,tok)(list(range(min(bs,len(ds)))));rw=bt.pop("row_weights").cuda();y=bt.pop("labels").cuda();bt={k:v.cuda() for k,v in bt.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16):loss=(F.binary_cross_entropy_with_logits(model(**bt).logits.squeeze(-1),y,reduction="none")*rw).sum()/rw.sum()
            loss.backward();torch.cuda.synchronize();pre.append({"batch":bs,"ok":True,"peak_gb":torch.cuda.max_memory_allocated()/2**30});physical=bs;model.zero_grad(set_to_none=True);break
        except torch.cuda.OutOfMemoryError:
            pre.append({"batch":bs,"ok":False});model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    if physical is None:raise RuntimeError("no safe batch")
    atomic_json(EXP/"runtime/oom_preflight.json",{"attempts":pre,"selected_physical_batch":physical})
    effective=max(512,physical);accum=math.ceil(effective/physical);gen=torch.Generator().manual_seed(SEED)
    loader=DataLoader(ds,batch_size=physical,shuffle=True,generator=gen,num_workers=4,pin_memory=True,collate_fn=Collator(ds,tok))
    opt=optimizer_for(model);steps=math.ceil(len(loader)/accum)*3;sched=get_linear_schedule_with_warmup(opt,int(.05*steps),steps);log=[]
    for epoch in range(1,4):
        model.train();opt.zero_grad(set_to_none=True);tot=0.;seen=0;t0=time.time()
        for step,bt in enumerate(loader):
            rw=bt.pop("row_weights").cuda(non_blocking=True);y=bt.pop("labels").cuda(non_blocking=True);bt={k:v.cuda(non_blocking=True) for k,v in bt.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16):raw=F.binary_cross_entropy_with_logits(model(**bt).logits.squeeze(-1),y,reduction="none");loss=(raw*rw).sum()/rw.sum()/accum
            loss.backward();tot+=float(loss.detach())*accum*len(y);seen+=len(y)
            if (step+1)%accum==0 or step+1==len(loader):torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sched.step();opt.zero_grad(set_to_none=True)
        sec=time.time()-t0;row={"epoch":epoch,"loss":tot/seen,"seconds":sec,"examples_per_second":seen/sec,"physical_batch":physical,"gradient_accumulation":accum,"effective_batch":physical*accum}
        log.append(row);save_model(model,tok,EXP/f"model/xlmr/epoch-{epoch}",{"epoch":epoch,"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"row":row})
        pd.DataFrame(log).to_csv(EXP/"runtime/training_log.csv",index=False);print("EPOCH",json.dumps(row),flush=True)
    atomic_json(EXP/"runtime/tokenization.json",{"rows":len(d),"truncated":trunc,"truncation_rate":trunc/len(d),"max_length":160})


def infer_new(input_path,checkpoint,output_path,batch=512):
    d=pd.read_parquet(input_path);model,tok=model_and_tokenizer(Path(checkpoint));model.cuda().eval();ids,trunc=encode_frame(d,tok,Path(output_path).with_suffix(".tokens.pt"));ds=EncodedDataset(ids);loader=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=4,pin_memory=True,collate_fn=Collator(ds,tok));out=[];t0=time.time()
    with torch.inference_mode():
        for bt in loader:
            bt={k:v.cuda(non_blocking=True) for k,v in bt.items()}
            with torch.autocast("cuda",dtype=torch.bfloat16):out.append(torch.sigmoid(model(**bt).logits.squeeze(-1).float()).cpu().numpy())
    atomic_parquet(pd.DataFrame({"xlmr_probability":np.concatenate(out).astype(np.float32)}),Path(output_path));atomic_json(Path(output_path).with_suffix(".runtime.json"),{"rows":len(d),"seconds":time.time()-t0,"truncated":trunc})


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("infer-old");p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--batch",type=int,default=512)
    sp.add_parser("infer-old-all");sp.add_parser("train")
    p=sp.add_parser("infer-new");p.add_argument("--input",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--batch",type=int,default=512)
    a=ap.parse_args()
    if a.cmd=="infer-old":infer_old(a.input,a.output,a.batch)
    elif a.cmd=="infer-old-all":infer_old_all()
    elif a.cmd=="train":train()
    else:infer_new(a.input,a.checkpoint,a.output,a.batch)


if __name__=="__main__":main()
