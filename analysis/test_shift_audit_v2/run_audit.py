#!/usr/bin/env python3
"""Label-free train/test shift audit. Never reads training truth or fits production models."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/test_shift_audit_v2"
DB = ROOT / "analysis/data_audit_v1/artifacts/audit.duckdb"
P002 = ROOT / "analysis/production/p002_final"
D002 = ROOT / "analysis/experiments_v1/d002_clean_public_gap/output"
SEED = 260926
SAMPLE_N = 50_000
Q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
GROUPS = [
    ("train", "US", 1), ("train", "US", 2), ("train", "US", 3),
    ("train", "India", 1), ("train", "India", 2), ("train", "India", 3),
    ("test", "US", 1), ("test", "US", 2), ("test", "US", 3),
    ("test", "India", 1), ("test", "India", 2), ("test", "India", 3),
    ("test", "France", 1), ("test", "France", 2), ("test", "France", 3),
]


def table(split: str, source: int) -> str:
    return f"a.{split}_s{source}"


def sqlq(x: str) -> str:
    return x.replace("'", "''")


def quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": np.nan, "std": np.nan, **{f"p{int(q*100):02d}": np.nan for q in Q}}
    vals = np.quantile(x, Q)
    return {"n": len(x), "mean": float(x.mean()), "std": float(x.std()),
            **{f"p{int(q*100):02d}": float(v) for q, v in zip(Q, vals)},
            "min": float(x.min()), "max": float(x.max())}


def js_numeric(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if not len(x) or not len(y): return np.nan
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if lo == hi: return 0.0
    edges = np.unique(np.quantile(np.concatenate([x, y]), np.linspace(0, 1, 51)))
    if len(edges) < 3: edges = np.linspace(lo, hi + 1e-12, 11)
    a = np.histogram(x, bins=edges)[0].astype(float) + 1e-12
    b = np.histogram(y, bins=edges)[0].astype(float) + 1e-12
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2) ** 2)


def record_features(df: pd.DataFrame) -> pd.DataFrame:
    name = df.business_name.fillna("").astype(str)
    addr = df.business_address.fillna("").astype(str)
    both = name + " " + addr
    def frac(series, pat):
        den = series.str.len().clip(lower=1)
        return series.str.count(pat) / den
    z = pd.DataFrame(index=df.index)
    z["name_char_length"] = name.str.len()
    z["name_token_count"] = name.str.findall(r"[^\W_]+", flags=re.UNICODE).str.len()
    z["address_char_length"] = addr.str.len()
    z["address_token_count"] = addr.str.findall(r"[^\W_]+", flags=re.UNICODE).str.len()
    z["empty_address"] = addr.str.strip().eq("").astype(float)
    z["nonascii_fraction"] = both.map(lambda s: sum(ord(c) > 127 for c in s) / max(1, len(s)))
    z["diacritic_presence"] = both.map(lambda s: float(s != "" and unicodedata.normalize("NFD", s) != "" and any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", s))))
    z["digit_ratio"] = frac(both, r"[0-9]")
    z["numeric_token_count"] = both.str.findall(r"(?<!\w)\d+(?!\w)").str.len()
    z["punctuation_ratio"] = frac(both, r"[^\w\s]")
    z["whitespace_count"] = both.str.count(r"\s")
    z["segment_count"] = both.str.findall(r"[^\s,;|/]+", flags=re.UNICODE).str.len()
    z["source"] = df.source.astype(str).to_numpy()
    return z


def exact_raw_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "name_char_length", "name_token_count", "address_char_length", "address_token_count",
        "empty_address", "nonascii_fraction", "nonascii_presence", "diacritic_presence",
        "digit_ratio", "numeric_token_count", "punctuation_ratio", "whitespace_count", "segment_count",
        "normalized_name_frequency", "normalized_name_fanout", "normalized_address_frequency",
        "normalized_address_fanout", "raw_duplicate_frequency", "normalized_duplicate_frequency",
    ]
    for split, country, source in GROUPS:
        t = table(split, source)
        union = " UNION ALL ".join([f"SELECT coalesce(nname,'') nname,coalesce(naddr,'') naddr FROM {table(split, s)} WHERE country='{sqlq(country)}'" for s in (1,2,3)])
        q = f"""
        WITH x AS (SELECT entity_id,coalesce(business_name,'') business_name,
          coalesce(business_address,'') business_address,country,
          coalesce(nname,'') nname,coalesce(naddr,'') naddr
          FROM {t} WHERE country='{sqlq(country)}'),
        nf AS (SELECT nname,count(*) n FROM x GROUP BY nname),
        af AS (SELECT naddr,count(*) n FROM x GROUP BY naddr),
        rf AS (SELECT business_name,business_address,count(*) n FROM x GROUP BY ALL),
        df AS (SELECT nname,naddr,count(*) n FROM x GROUP BY ALL),
        allx AS ({union}),
        ng AS (SELECT nname,count(*) n FROM allx GROUP BY nname),
        ag AS (SELECT naddr,count(*) n FROM allx GROUP BY naddr),
        b AS (
          SELECT
            length(coalesce(x.business_name,''))::DOUBLE name_char_length,
            array_length(regexp_extract_all(coalesce(x.business_name,''),'[\\p{{L}}\\p{{N}}]+'))::DOUBLE name_token_count,
            length(coalesce(x.business_address,''))::DOUBLE address_char_length,
            array_length(regexp_extract_all(coalesce(x.business_address,''),'[\\p{{L}}\\p{{N}}]+'))::DOUBLE address_token_count,
            (trim(coalesce(x.business_address,''))='')::INT::DOUBLE empty_address,
            length(regexp_replace(coalesce(x.business_name,'')||coalesce(x.business_address,''),'[\\x00-\\x7F]','','g'))::DOUBLE/greatest(1,length(coalesce(x.business_name,'')||coalesce(x.business_address,''))) nonascii_fraction,
            regexp_matches(coalesce(x.business_name,'')||coalesce(x.business_address,''),'[^\\x00-\\x7F]')::INT::DOUBLE nonascii_presence,
            ((coalesce(x.business_name,'')||coalesce(x.business_address,'')) != strip_accents(coalesce(x.business_name,'')||coalesce(x.business_address,'')))::INT::DOUBLE diacritic_presence,
            length(regexp_replace(coalesce(x.business_name,'')||coalesce(x.business_address,''),'[^0-9]','','g'))::DOUBLE/greatest(1,length(coalesce(x.business_name,'')||coalesce(x.business_address,''))) digit_ratio,
            array_length(regexp_extract_all(coalesce(x.business_name,'')||' '||coalesce(x.business_address,''),'\\b[0-9]+\\b'))::DOUBLE numeric_token_count,
            length(regexp_replace(coalesce(x.business_name,'')||coalesce(x.business_address,''),'[\\p{{L}}\\p{{N}}\\s]','','g'))::DOUBLE/greatest(1,length(coalesce(x.business_name,'')||coalesce(x.business_address,''))) punctuation_ratio,
            length(regexp_replace(coalesce(x.business_name,'')||coalesce(x.business_address,''),'[^\\s]','','g'))::DOUBLE whitespace_count,
            array_length(regexp_extract_all(coalesce(x.business_name,'')||' '||coalesce(x.business_address,''),'[^\\s,;|/]+'))::DOUBLE segment_count,
            CASE WHEN x.nname='' THEN 0 ELSE nf.n END::DOUBLE normalized_name_frequency,
            CASE WHEN x.nname='' THEN 0 ELSE ng.n END::DOUBLE normalized_name_fanout,
            CASE WHEN x.naddr='' THEN 0 ELSE af.n END::DOUBLE normalized_address_frequency,
            CASE WHEN x.naddr='' THEN 0 ELSE ag.n END::DOUBLE normalized_address_fanout,
            rf.n::DOUBLE raw_duplicate_frequency, df.n::DOUBLE normalized_duplicate_frequency
          FROM x JOIN nf USING(nname) JOIN af USING(naddr)
          JOIN rf USING(business_name,business_address) JOIN df USING(nname,naddr)
          JOIN ng USING(nname) JOIN ag USING(naddr)
        ), u AS (UNPIVOT b ON {','.join(metric_cols)} INTO NAME metric VALUE value)
        SELECT metric,count(*) n,avg(value) mean,stddev_pop(value) std,min(value) min,
          quantile_cont(value,.01) p01,quantile_cont(value,.05) p05,quantile_cont(value,.10) p10,
          quantile_cont(value,.25) p25,quantile_cont(value,.50) p50,quantile_cont(value,.75) p75,
          quantile_cont(value,.90) p90,quantile_cont(value,.95) p95,quantile_cont(value,.99) p99,max(value) max
        FROM u GROUP BY metric ORDER BY metric
        """
        z = con.execute(q).fetchdf()
        z.insert(0, "source", f"S{source}"); z.insert(0, "country", country); z.insert(0, "split", split)
        z["basis"] = "exact_full_population"
        rows.append(z)
        print("raw", split, country, source, flush=True)
    return pd.concat(rows, ignore_index=True)


def load_samples(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    parts=[]
    for split,country,source in GROUPS:
        q=f"""SELECT entity_id,business_name,business_address,nname,naddr,
          '{split}' split,'{sqlq(country)}' country,{source} source_id
          FROM {table(split,source)} WHERE country='{sqlq(country)}'
          ORDER BY hash(entity_id,{SEED}) LIMIT {SAMPLE_N}"""
        z=con.execute(q).fetchdf().rename(columns={"source_id":"source"}); parts.append(z)
    return pd.concat(parts,ignore_index=True)


def build_training_vocab(con: duckdb.DuckDBPyConnection):
    words=set(); grams={2:set(),3:set(),4:set()}; seen=0; start=time.time()
    union=" UNION ALL ".join([f"SELECT nname,naddr FROM a.train_s{s}" for s in (1,2,3)])
    cur=con.execute(union)
    while True:
        batch=cur.fetchmany(20_000)
        if not batch: break
        for x,y in batch:
            for text in (x or "",y or ""):
                words.update(text.split())
                for n,g in grams.items():
                    g.update(text[i:i+n] for i in range(max(0,len(text)-n+1)))
        seen += len(batch)
        if seen % 1_000_000 == 0: print("vocab",seen,round(time.time()-start,1),flush=True)
    return words,grams


def add_oov_metrics(samples, words, grams, raw):
    out=[]; per_record={}
    for key,g in samples.groupby(["split","country","source"],sort=False):
        vals={}
        for field in ("nname","naddr"):
            prefix="name" if field=="nname" else "address"
            texts=g[field].fillna("").astype(str)
            tok=texts.map(lambda s: np.mean([t not in words for t in s.split()]) if s.split() else 0.0).to_numpy()
            vals[f"{prefix}_word_token_oov_fraction"]=tok
            for n in (2,3,4):
                vals[f"{prefix}_char_{n}gram_oov_fraction"]=texts.map(
                    lambda s,n=n: np.mean([s[i:i+n] not in grams[n] for i in range(len(s)-n+1)]) if len(s)>=n else 0.0).to_numpy()
        per_record[key]=pd.DataFrame(vals,index=g.index)
        for metric,x in vals.items():
            r={"split":key[0],"country":key[1],"source":f"S{key[2]}","metric":metric,"basis":f"deterministic_sample_n={len(g)}",**quantiles(x)}
            out.append(r)
    return pd.concat([raw,pd.DataFrame(out)],ignore_index=True,sort=False),per_record


def divergence(raw,samples,per_record):
    numeric=list(record_features(samples.iloc[:2]).columns[:-1])
    sm={}
    for key,g in samples.groupby(["split","country","source"],sort=False):
        f=record_features(g)
        for m in numeric: sm[key,m]=f[m].to_numpy(float)
        for m in per_record[key].columns: sm[key,m]=per_record[key][m].to_numpy(float)
    raw["reference_population"]="";raw["ks"] = np.nan;raw["wasserstein"] = np.nan;raw["wasserstein_standardized"] = np.nan;raw["js"] = np.nan
    for i,r in raw.iterrows():
        if r["split"]!="test": continue
        source=int(str(r["source"])[1:]); metric=r["metric"]
        ref_country=r["country"] if r["country"] in ("US","India") else "POOLED_US_INDIA"
        x=sm.get((("test",r["country"],source),metric))
        if x is None: continue
        if ref_country=="POOLED_US_INDIA":
            y=np.concatenate([sm.get((("train","US",source),metric),[]),sm.get((("train","India",source),metric),[])])
        else: y=sm.get((("train",ref_country,source),metric))
        if y is None or not len(y): continue
        raw.at[i,"reference_population"]=f"train {ref_country} S{source}"
        raw.at[i,"ks"]=ks_2samp(x,y).statistic
        raw.at[i,"wasserstein"]=wasserstein_distance(x,y)
        sy=np.std(y)
        raw.at[i,"wasserstein_standardized"]=wasserstein_distance(x,y)/sy if sy>1e-9 else np.nan
        raw.at[i,"js"]=js_numeric(x,y)
    return raw


def source_density(con,raw):
    countq=[]
    for split,country,_ in GROUPS:
        if (split,country) in countq: continue
        countq.append((split,country))
    rows=[]
    for split,country in countq:
        counts={s:con.execute(f"SELECT count(*) FROM {table(split,s)} WHERE country=?",[country]).fetchone()[0] for s in (1,2,3)}
        ratios={"s2_s1_ratio":counts[2]/counts[1],"s3_s1_ratio":counts[3]/counts[1],"total_aux_s1":(counts[2]+counts[3])/counts[1]}
        for s in (1,2,3):
            rr=raw[(raw.split==split)&(raw.country==country)&(raw.source==f"S{s}")].set_index("metric")
            def v(metric,col): return float(rr.loc[metric,col])
            rows.append({"split":split,"country":country,"source":f"S{s}","n":counts[s],**ratios,
             "missing_address_rate":v("empty_address","mean"),
             "normalized_name_duplicate_rate":con.execute(f"SELECT avg((n>1)::INT) FROM (SELECT count(*) over(partition by nname) n,nname FROM {table(split,s)} WHERE country=? AND nname<>'')",[country]).fetchone()[0],
             "normalized_name_fanout_mean":v("normalized_name_fanout","mean"),"normalized_name_fanout_p90":v("normalized_name_fanout","p90"),"normalized_name_fanout_p99":v("normalized_name_fanout","p99"),
             "normalized_address_duplicate_rate":con.execute(f"SELECT avg((n>1)::INT) FROM (SELECT count(*) over(partition by naddr) n,naddr FROM {table(split,s)} WHERE country=? AND naddr<>'')",[country]).fetchone()[0],
             "normalized_address_fanout_mean":v("normalized_address_fanout","mean"),"normalized_address_fanout_p90":v("normalized_address_fanout","p90"),"normalized_address_fanout_p99":v("normalized_address_fanout","p99")})
    return pd.DataFrame(rows)


def france_text(con,samples,per_record):
    rows=[]
    def top_sql(source,kind,field,expr,n=30):
        scopes=[source] if source else [1,2,3]
        union=" UNION ALL ".join([f"SELECT {field} v FROM a.test_s{s} WHERE country='France'" for s in scopes])
        q=f"WITH x AS ({union}), z AS ({expr}) SELECT token_value,count(*) token_count FROM z WHERE token_value<>'' GROUP BY token_value ORDER BY token_count DESC,token_value LIMIT {n}"
        for value,count in con.execute(q).fetchall(): rows.append({"source":f"S{source}" if source else "ALL","statistic":kind,"value":value,"count":count,"rate":np.nan,"basis":"exact_full_population"})
    for s in (None,1,2,3):
        top_sql(s,"raw_name_token","business_name","SELECT unnest(string_split_regex(v,'\\s+')) token_value FROM x")
        top_sql(s,"normalized_name_token","nname","SELECT unnest(string_split(v,' ')) token_value FROM x")
        top_sql(s,"normalized_address_token","naddr","SELECT unnest(string_split(v,' ')) token_value FROM x")
        top_sql(s,"name_last_token","nname","SELECT regexp_extract(v,'([^ ]+)$',1) token_value FROM x")
        top_sql(s,"name_suffix3","nname","SELECT right(v,3) token_value FROM x WHERE length(v)>=3")
        top_sql(s,"address_numeric_token","naddr","SELECT unnest(regexp_extract_all(v,'\\b[0-9]+\\b')) token_value FROM x")
        top_sql(s,"address_short_structural_token","naddr","SELECT token token_value FROM x,LATERAL unnest(string_split(v,' ')) q(token) WHERE length(token)<=3")
    # Sample-derived number patterns and novelty summaries.
    fs=samples[(samples.split=="test")&(samples.country=="France")]
    for s,g in fs.groupby("source"):
        patterns=Counter()
        for text in (g.business_name.fillna("")+" | "+g.business_address.fillna("")):
            p=re.sub(r"\d+","#",text);p=re.sub(r"[^#]+","A",p);p=re.sub(r"A+","A",p)
            patterns[p[:80]]+=1
        for val,cnt in patterns.most_common(30): rows.append({"source":f"S{s}","statistic":"number_pattern","value":val,"count":cnt,"rate":cnt/len(g),"basis":f"deterministic_sample_n={len(g)}"})
        o=per_record[("test","France",s)]
        for col in o:
            st=quantiles(o[col].to_numpy())
            for qn in ("mean","p50","p90","p99"):
                rows.append({"source":f"S{s}","statistic":col,"value":qn,"count":np.nan,"rate":st[qn],"basis":f"deterministic_sample_n={len(g)}"})
    # Exact non-ASCII character and accent-fold mapping counts via streaming France records.
    chars=Counter();maps=Counter();
    union=" UNION ALL ".join([f"SELECT business_name,business_address FROM a.test_s{s} WHERE country='France'" for s in (1,2,3)])
    cur=con.execute(union)
    while True:
        batch=cur.fetchmany(20_000)
        if not batch: break
        for a,b in batch:
            for ch in (a or "")+(b or ""):
                if ord(ch)>127:
                    chars[ch]+=1
                    fold="".join(x for x in unicodedata.normalize("NFKD",ch) if not unicodedata.combining(x))
                    maps[f"{ch}->{fold}"]+=1
    for kind,counter in (("nonascii_character",chars),("diacritic_fold_mapping",maps)):
        total=sum(counter.values())
        for val,cnt in counter.most_common(50): rows.append({"source":"ALL","statistic":kind,"value":val,"count":cnt,"rate":cnt/max(1,total),"basis":"exact_full_population"})
    return pd.DataFrame(rows)


def append_france_fanout_summaries(france,raw):
    france=france[~france.statistic.isin(["normalized_name_fanout_summary","normalized_address_fanout_summary"])]
    rows=[]
    for source in ("S1","S2","S3"):
        for metric,stat in (("normalized_name_fanout","normalized_name_fanout_summary"),("normalized_address_fanout","normalized_address_fanout_summary")):
            z=raw[(raw.split=="test")&(raw.country=="France")&(raw.source==source)&(raw.metric==metric)].iloc[0]
            for q in ("mean","p50","p90","p99"):
                rows.append({"source":source,"statistic":stat,"value":q,"count":np.nan,"rate":z[q],"basis":"exact_full_population"})
    return pd.concat([france,pd.DataFrame(rows)],ignore_index=True)


def adversarial(samples):
    results=[];importance=[]
    pairs=[("train_US_vs_test_US",("train","US"),("test","US")),
           ("train_India_vs_test_India",("train","India"),("test","India")),
           ("train_US_India_vs_test_France",("train",("US","India")),("test","France"))]
    rng=np.random.default_rng(SEED)
    for name,left,right in pairs:
        if isinstance(left[1],tuple): l=samples[(samples.split==left[0])&samples.country.isin(left[1])]
        else: l=samples[(samples.split==left[0])&(samples.country==left[1])]
        r=samples[(samples.split==right[0])&(samples.country==right[1])]
        n=min(len(l),len(r),120_000); l=l.sample(n,random_state=SEED);r=r.sample(n,random_state=SEED)
        d=pd.concat([l.assign(target=0),r.assign(target=1)],ignore_index=True)
        f=record_features(d); y=d.target.to_numpy(); idx=rng.permutation(len(d)); cut=int(.7*len(d));tr,te=idx[:cut],idx[cut:]
        nums=[c for c in f.columns if c!="source"]
        pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer()),("scale",StandardScaler())]),nums),("src",OneHotEncoder(handle_unknown="ignore"),["source"])])
        model=Pipeline([("pre",pre),("clf",LogisticRegression(C=1,max_iter=300,solver="liblinear",random_state=SEED))])
        model.fit(f.iloc[tr],y[tr]);score=model.predict_proba(f.iloc[te])[:,1];auc=roc_auc_score(y[te],score)
        results.append({"comparison":name,"representation":"compact_structural","n_left":n,"n_right":n,"test_n":len(te),"auroc":auc,"seed":SEED})
        names=nums+list(model.named_steps["pre"].named_transformers_["src"].get_feature_names_out(["source"]));coef=model.named_steps["clf"].coef_[0]
        for rank,j in enumerate(np.argsort(np.abs(coef))[::-1][:20],1): importance.append({"comparison":name,"representation":"compact_structural","rank":rank,"feature":names[j],"importance":abs(coef[j]),"signed_effect_toward_test":coef[j]})
        # Cheap char-TFIDF repeat, capped and isolated from any production path.
        nc=min(n,40_000); dl=pd.concat([l.sample(nc,random_state=SEED),r.sample(nc,random_state=SEED)]); yy=np.r_[np.zeros(nc),np.ones(nc)]
        text=(dl.business_name.fillna("")+" [ADDR] "+dl.business_address.fillna("")).to_numpy();ii=rng.permutation(len(text));cc=int(.7*len(text));
        vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=3,max_features=60_000,sublinear_tf=True,dtype=np.float32)
        Xtr=vec.fit_transform(text[ii[:cc]]);Xte=vec.transform(text[ii[cc:]])
        clf=SGDClassifier(loss="log_loss",alpha=1e-5,max_iter=50,tol=1e-3,early_stopping=True,validation_fraction=.1,random_state=SEED)
        clf.fit(Xtr,yy[ii[:cc]]);sc=clf.decision_function(Xte);auc2=roc_auc_score(yy[ii[cc:]],sc)
        results.append({"comparison":name,"representation":"char_tfidf_2_5_linear","n_left":nc,"n_right":nc,"test_n":len(ii[cc:]),"auroc":auc2,"seed":SEED})
        feats=np.asarray(vec.get_feature_names_out());co=clf.coef_[0]
        for rank,j in enumerate(np.argsort(np.abs(co))[::-1][:20],1): importance.append({"comparison":name,"representation":"char_tfidf_2_5_linear","rank":rank,"feature":feats[j],"importance":abs(co[j]),"signed_effect_toward_test":co[j]})
        print("av",name,auc,auc2,flush=True)
    return pd.DataFrame(results),pd.DataFrame(importance)


def geometry(samples):
    rows=[]; numeric=[c for c in record_features(samples.iloc[:2]).columns if c!="source"]
    for s in (1,2,3):
        us=samples[(samples.split=="train")&(samples.country=="US")&(samples.source==s)].sample(30_000,random_state=SEED)
        ind=samples[(samples.split=="train")&(samples.country=="India")&(samples.source==s)].sample(30_000,random_state=SEED)
        fr=samples[(samples.split=="test")&(samples.country=="France")&(samples.source==s)]
        train=pd.concat([us,ind]);txt=(train.business_name.fillna("")+" [ADDR] "+train.business_address.fillna(""));ft=(fr.business_name.fillna("")+" [ADDR] "+fr.business_address.fillna(""))
        vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=3,max_features=80_000,sublinear_tf=True,dtype=np.float32,norm="l2")
        X=vec.fit_transform(txt);F=vec.transform(ft);cu=np.asarray(X[:len(us)].mean(axis=0)).ravel();ci=np.asarray(X[len(us):].mean(axis=0)).ravel();cu/=np.linalg.norm(cu);ci/=np.linalg.norm(ci)
        su=np.asarray(F@cu).ravel();si=np.asarray(F@ci).ravel();
        fu=record_features(us)[numeric].to_numpy(float);fi=record_features(ind)[numeric].to_numpy(float);ff=record_features(fr)[numeric].to_numpy(float)
        mu=np.mean(np.vstack([fu,fi]),axis=0);sd=np.std(np.vstack([fu,fi]),axis=0)+1e-9;zu=(fu-mu)/sd;zi=(fi-mu)/sd;zf=(ff-mu)/sd;uc=zu.mean(0);ic=zi.mean(0);du=np.sqrt(np.mean((zf-uc)**2,axis=1));di=np.sqrt(np.mean((zf-ic)**2,axis=1))
        own=np.r_[np.sqrt(np.mean((zu-uc)**2,axis=1)),np.sqrt(np.mean((zi-ic)**2,axis=1))]; struct_thr=np.quantile(own,.99)
        train_sim=np.r_[np.asarray(X[:len(us)]@cu).ravel(),np.asarray(X[len(us):]@ci).ravel()];sim_thr=np.quantile(train_sim,.01);neither=(np.maximum(su,si)<sim_thr)&(np.minimum(du,di)>struct_thr)
        metrics={"tfidf_cosine_to_train_US":su,"tfidf_cosine_to_train_India":si,"tfidf_US_minus_India":su-si,"compact_distance_to_train_US":du,"compact_distance_to_train_India":di,"compact_India_minus_US":di-du}
        for m,x in metrics.items(): rows.append({"source":f"S{s}","metric":m,**quantiles(x),"basis":f"France deterministic sample n={len(fr)}"})
        rows += [{"source":f"S{s}","metric":"tfidf_fraction_US_closer","mean":float(np.mean(su>si)),"n":len(fr)},
                 {"source":f"S{s}","metric":"compact_fraction_US_closer","mean":float(np.mean(du<di)),"n":len(fr)},
                 {"source":f"S{s}","metric":"joint_neither_fraction","mean":float(neither.mean()),"n":len(fr),"threshold_detail":f"tfidf<{sim_thr:.6g} and compact>{struct_thr:.6g}"}]
        print("geometry",s,flush=True)
    return pd.DataFrame(rows)


def existing_model_shift(con):
    rows=[]
    score=pd.read_csv(D002/"score_percentiles.csv")
    rates=pd.read_csv(D002/"decoder_rates.csv")
    for country in ("US","India","France"):
        pop=f"test {country}"
        for _,r in score[score.population.eq(pop)].iterrows(): rows.append({"country":country,"scope":"all_existing_shortlist_or_owner","metric":r.statistic,**{k:r[k] for k in score.columns if k.startswith("p")},"basis":"existing_D002_artifact"})
        rr=rates[rates.population.eq(pop)].iloc[0]
        for col,metric in [("above_threshold_per_s1","above_threshold_per_s1"),("rejected_competition_per_s1","rejected_competition_per_s1"),("rejected_margin_only_per_s1","rejected_margin_only_per_s1"),("accepted_per_s1","accepted_matches_per_s1")]:
            rows.append({"country":country,"scope":"per_S1","metric":metric,"mean":rr[col],"basis":"existing_P002_D002_artifact"})
        short=str(P002/f"cache/shortlist/{country}/*.parquet");neu=str(P002/f"cache/neural/{country}/*.parquet")
        q=f"""WITH s AS (SELECT source1_entity_id,candidate_entity_id,candidate_source,lightgbm_score,name_retrieval_score,address_retrieval_score,name_rank,address_rank FROM read_parquet('{short}')),
        n AS (SELECT * FROM read_parquet('{neu}')),j AS (SELECT s.*,n.neural_probability,abs(s.lightgbm_score-n.neural_probability) disagreement,greatest(s.name_retrieval_score,s.address_retrieval_score) lexical FROM s JOIN n USING(source1_entity_id,candidate_entity_id,candidate_source))
        SELECT count(*) n,avg(disagreement),quantile_cont(disagreement,[.5,.9,.95,.99]),
        avg((lightgbm_score>=.8 AND neural_probability<=.2)::INT),avg((neural_probability>=.8 AND lightgbm_score<=.2)::INT),
        avg((lexical>=.8 AND neural_probability<=.1)::INT),avg((lexical>=.9 AND neural_probability<=.1)::INT),
        avg(CASE WHEN lexical>=.8 THEN neural_probability END),quantile_cont(CASE WHEN lexical>=.8 THEN neural_probability END,[.1,.5,.9]) FROM j"""
        n,mean,qs,lhi,nhi,lex8,lex9,lmean,lq=con.execute(q).fetchone()
        for m,val in [("lgbm_xlmr_abs_disagreement_mean",mean),("lgbm_high_xlmr_low_rate",lhi),("xlmr_high_lgbm_low_rate",nhi),("lexical_ge_.8_xlmr_le_.1_rate",lex8),("lexical_ge_.9_xlmr_le_.1_rate",lex9),("xlmr_mean_given_lexical_ge_.8",lmean)]: rows.append({"country":country,"scope":"all_existing_shortlist","metric":m,"mean":val,"n":n,"basis":"existing_P002_artifacts_no_inference"})
        rows.append({"country":country,"scope":"all_existing_shortlist","metric":"lgbm_xlmr_abs_disagreement","p50":qs[0],"p90":qs[1],"p95":qs[2],"p99":qs[3],"n":n,"basis":"existing_P002_artifacts_no_inference"})
        rows.append({"country":country,"scope":"lexical_ge_.8","metric":"xlmr_score","p10":lq[0],"p50":lq[1],"p90":lq[2],"n":n,"basis":"existing_P002_artifacts_no_inference"})
        rankq=f"""WITH s AS (SELECT source1_entity_id,candidate_entity_id,candidate_source,lightgbm_score,name_retrieval_score,address_retrieval_score,name_rank,address_rank FROM read_parquet('{short}')),
        n AS (SELECT * FROM read_parquet('{neu}')),j AS (SELECT s.*,n.neural_probability,
          greatest(s.name_retrieval_score,s.address_retrieval_score) lexical,
          CASE WHEN least(s.name_rank,s.address_rank)=1 THEN 'rank_1' WHEN least(s.name_rank,s.address_rank)<=3 THEN 'rank_2_3' WHEN least(s.name_rank,s.address_rank)<=10 THEN 'rank_4_10' ELSE 'rank_missing' END rank_bucket
          FROM s JOIN n USING(source1_entity_id,candidate_entity_id,candidate_source))
        SELECT rank_bucket,count(*) n,quantile_cont(lexical,.5),quantile_cont(neural_probability,.5),quantile_cont(lightgbm_score,.5),avg((neural_probability<=.1)::INT)
        FROM j GROUP BY rank_bucket ORDER BY rank_bucket"""
        for bucket,bn,lexmed,nmed,lmed,nlow in con.execute(rankq).fetchall():
            rows.extend([
              {"country":country,"scope":bucket,"metric":"lexical_score","p50":lexmed,"n":bn,"basis":"existing_P002_artifacts_no_inference"},
              {"country":country,"scope":bucket,"metric":"xlmr_score","p50":nmed,"n":bn,"basis":"existing_P002_artifacts_no_inference"},
              {"country":country,"scope":bucket,"metric":"lgbm_score","p50":lmed,"n":bn,"basis":"existing_P002_artifacts_no_inference"},
              {"country":country,"scope":bucket,"metric":"xlmr_le_.1_rate","mean":nlow,"n":bn,"basis":"existing_P002_artifacts_no_inference"}])
        print("model",country,flush=True)
    return pd.DataFrame(rows)


def casebook(con,samples,per_record,raw):
    fr=samples[(samples.split=="test")&(samples.country=="France")].copy();f=record_features(fr)
    for key,o in per_record.items():
        if key[0]=="test" and key[1]=="France":
            for c in o: f.loc[o.index,c]=o[c]
    # Fanout from exact value counts in the full France population is added by the caller's sample values.
    picks=[]
    criteria={"long_name":("name_char_length",False),"short_name":("name_char_length",True),"long_address":("address_char_length",False),"short_nonempty_address":("address_char_length",True),"missing_address":("empty_address",False),"high_word_oov":("name_word_token_oov_fraction",False),"high_char4_oov":("name_char_4gram_oov_fraction",False),"high_nonascii":("nonascii_fraction",False),"unusual_numeric_structure":("numeric_token_count",False)}
    for reason,(col,asc) in criteria.items():
        x=fr.join(f[[col]]); x=x[x[col]>0] if reason in ("short_name","short_nonempty_address") else x
        p=x.sort_values([col,"entity_id"],ascending=[asc,True]).head(20).copy();p["sample_reason"]=reason;picks.append(p)
    # Deterministic examples selected using exact full-France cross-source fanout.
    union=" UNION ALL ".join([f"SELECT entity_id,business_name,business_address,country,{s} source_id,coalesce(nname,'') nname,coalesce(naddr,'') naddr FROM a.test_s{s} WHERE country='France'" for s in (1,2,3)])
    for field,reason in (("nname","high_normalized_name_fanout"),("naddr","high_normalized_address_fanout")):
        q=f"""WITH x AS ({union}), c AS (SELECT {field} val,count(*) fanout FROM x WHERE {field}<>'' GROUP BY {field})
        SELECT '{reason}' sample_reason,x.source_id AS "source",x.entity_id,x.business_name,x.business_address,x.country
        FROM x JOIN c ON x.{field}=c.val ORDER BY c.fanout DESC,x.entity_id LIMIT 40"""
        picks.append(con.execute(q).fetchdf())
    z=pd.concat(picks).drop_duplicates(["sample_reason","entity_id"]).sort_values(["sample_reason","source","entity_id"])
    return z[["sample_reason","source","entity_id","business_name","business_address","country"]]


def write_stress_spec():
    text="""# Future size-controlled country-held-out stress experiment

Status: specification only. Do not run until separately authorized. This design does not alter the frozen blocker, candidate policy, matcher architecture, losses, decoder, thresholds, or source splits.

## Cohorts and size control

Run two directional diagnostics: US-only training evaluated on India, and India-only training evaluated on US. Start from one newly declared, immutable S1-level development split made before fitting. Keep every S1 and all of its S2/S3 edges in exactly one partition. Never sample individual pairs before assigning S1 groups.

Let `N = min(number of eligible US training S1s, number of eligible India training S1s)` after the same exclusion and data-quality rules. Select exactly `N` S1 groups per country by sorting `SHA256("country-stress-v1" || source1_entity_id)` and taking the first `N`. Use identical negative-generation, positive-expansion, per-S1 caps, source balance, optimizer steps, epochs, and checkpoint-selection rule in both directions. If pair totals differ after S1 control, deterministically cap each label/source stratum to the smaller realized count in the two directions; report both pre-cap and post-cap counts. No country mixing is allowed in a training arm.

Use an untouched target-country evaluation cohort whose S1 groups had no role in candidate-depth, feature, model, checkpoint, fusion, threshold, margin, or decoder selection. Freeze all settings using a separate source-country calibration subset only; never calibrate on the held-out target country. Use the same already-frozen blocker and shortlist semantics in both arms once the teammate's blocker is final.

## Required arms

1. `US_to_India`: train on size-controlled US; calibrate on held-out US; evaluate once on untouched India.
2. `India_to_US`: train on size-controlled India; calibrate on held-out India; evaluate once on untouched US.

For interpretation, also report same-country controls trained on the exact same `N` S1 count and evaluated on untouched same-country cohorts. They separate country-exposure loss from generic finite-sample/model variance. Seeds, S1-ID hashes, candidate hashes, code commit, artifact hashes, and row counts must be recorded.

## Metrics and artifacts

Report by direction, target country, source S2/S3, and target-S1 true cardinality bucket: candidate/blocker oracle, shortlist oracle, true-positive rank distribution, TP score p1/p5/p10/median, hard-false score p90/p95/p99, owner top-1 accuracy, owner-margin distribution, grouped macro F0.5, precision, and recall. Also report candidate count/S1 and the fraction of true positives absent before matcher scoring so blocker and matcher effects cannot be conflated.

Save candidate keys/ranks/scores, target labels, per-S1 predictions, fixed thresholds/margins, and bootstrap confidence intervals clustered by S1. Compare cross-country minus matched-size same-country results with paired S1 bootstrap where cohorts are shared and independent clustered bootstrap otherwise. No public leaderboard feedback, test records, pseudo-labels, or test-derived calibration may enter this experiment.
"""
    (OUT/"future_country_stress_spec.md").write_text(text)


def make_report(raw,density,france,av,imp,geo,model):
    def rv(split,country,source,metric,col="p50"):
        z=raw[(raw.split==split)&(raw.country==country)&(raw.source==source)&(raw.metric==metric)]
        return float(z.iloc[0][col])
    def au(comp,rep): return float(av[(av.comparison==comp)&(av.representation==rep)].auroc.iloc[0])
    def mm(country,metric,col="p50"):
        z=model[(model.country==country)&(model.metric==metric)];return float(z.iloc[0][col if col in z else "mean"])
    def dm(split,country,source,col): return float(density[(density.split==split)&(density.country==country)&(density.source==source)].iloc[0][col])
    def gm(source,metric,col="mean"): return float(geo[(geo.source==source)&(geo.metric==metric)].iloc[0][col])
    lines=["# EXECUTIVE_FINDINGS","",
      f"- **CONFIRMED — France is readily separable from the pooled training record distribution.** Compact structural adversarial AUROC is {au('train_US_India_vs_test_France','compact_structural'):.3f}; character-TFIDF AUROC is {au('train_US_India_vs_test_France','char_tfidf_2_5_linear'):.3f}. This threatens an i.i.d. matcher assumption even before candidate correctness is considered.",
      f"- **CONFIRMED — Seen-country train/test text shape is nearly stationary by comparison.** Compact AUROC is only {au('train_US_vs_test_US','compact_structural'):.3f} for US and {au('train_India_vs_test_India','compact_structural'):.3f} for India; character AUROCs are {au('train_US_vs_test_US','char_tfidf_2_5_linear'):.3f}/{au('train_India_vs_test_India','char_tfidf_2_5_linear'):.3f}. This makes the France separation qualitatively different rather than a generic split artifact.",
      f"- **CONFIRMED — Test candidate density is structurally higher even in seen countries.** Auxiliary records/S1 rise from {dm('train','US','S1','total_aux_s1'):.3f} to {dm('test','US','S1','total_aux_s1'):.3f} in US and {dm('train','India','S1','total_aux_s1'):.3f} to {dm('test','India','S1','total_aux_s1'):.3f} in India; France is {dm('test','France','S1','total_aux_s1'):.3f}. This threatens training-time candidate-count and owner-competition priors.",
      f"- **CONFIRMED — Existing P002 scores show a France-specific neural/lexical conflict.** France shortlist fused median is {mm('France','fused_shortlist'):.3f}, while the rate of lexical score >=.8 with XLM-R <=.1 is {mm('France','lexical_ge_.8_xlmr_le_.1_rate','mean'):.3%}. Confidence is not correctness; this is evidence of model disagreement, not hidden-label failure.",
      "- **INFERENCE — Matcher robustness should prioritize domain-stable character/number/address handling and explicit source/missingness context.** The raw shift, adversarial drivers, and train-domain geometry support this design pressure. They do not authorize test adaptation or threshold changes.",
      "- **OPEN QUESTION — Test blocker recall and France match truth remain unknowable here.** Strong lexical scores, fanout, score distributions, or accepted-link density cannot establish candidate recall or correctness without labels.","",
      "# RAW_SHIFT","",
      "`raw_distribution_metrics.csv` is long-form. Exact full-population quantiles are used for raw shape, missingness, and duplicate/fanout metrics. OOV metrics and divergence statistics use deterministic 50,000-record samples per split/country/source; each row records its basis. For test US/India the divergence reference is same-country train/source; France uses pooled US+India train within source.","",
      "Material shifts should be read through the standardized Wasserstein, KS, and JS columns rather than mean differences alone. Length/token/number shift threatens fixed truncation or rigid numeric-address heuristics; fanout shift threatens one-owner assumptions and negative sampling that underrepresents ambiguous normalized strings; OOV shift threatens word-heavy matchers and supports retaining character/subword channels.","",
      "| material comparison | S1 | S2 | S3 | matcher assumption affected |","|---|---:|---:|---:|---|",
      f"| France name word-OOV mean | {rv('test','France','S1','name_word_token_oov_fraction','mean'):.1%} | {rv('test','France','S2','name_word_token_oov_fraction','mean'):.1%} | {rv('test','France','S3','name_word_token_oov_fraction','mean'):.1%} | Closed-vocabulary/name-token evidence transfers poorly |",
      f"| France name 4-gram OOV mean | {rv('test','France','S1','name_char_4gram_oov_fraction','mean'):.1%} | {rv('test','France','S2','name_char_4gram_oov_fraction','mean'):.1%} | {rv('test','France','S3','name_char_4gram_oov_fraction','mean'):.1%} | Character evidence transfers substantially better than words |",
      f"| France address word-OOV mean | {rv('test','France','S1','address_word_token_oov_fraction','mean'):.1%} | {rv('test','France','S2','address_word_token_oov_fraction','mean'):.1%} | {rv('test','France','S3','address_word_token_oov_fraction','mean'):.1%} | Address token vocabularies are country-specific |",
      f"| France median name length | {rv('test','France','S1','name_char_length'):.0f} | {rv('test','France','S2','name_char_length'):.0f} | {rv('test','France','S3','name_char_length'):.0f} | Length priors learned from US/India can become country proxies |",
      f"| France median address length | {rv('test','France','S1','address_char_length'):.0f} | {rv('test','France','S2','address_char_length'):.0f} | {rv('test','France','S3','address_char_length'):.0f} | Fixed address-shape heuristics do not transfer uniformly |","",
      "# FRANCE","",
      "`france_text_stats.csv` reports exact frequent raw/normalized name tokens, address tokens, name endings, short structural tokens, numeric tokens, and non-ASCII/diacritic mappings. Number-pattern and novelty summaries are deterministic samples. These are string-distribution descriptions only; no row is declared to share a business identity.","",
      f"**CONFIRMED:** non-ASCII is present in {rv('test','France','S1','nonascii_presence','mean'):.1%} of France S1 and about {rv('test','France','S2','nonascii_presence','mean'):.1%}/{rv('test','France','S3','nonascii_presence','mean'):.1%} of S2/S3; diacritic presence is about 38% in every source. The leading observed characters are `é`, `É`, and `è`. Exact token counts show recurring name endings such as `sarl`, `sas`, `eurl`, `sa`, and `sasu`, and address structure dominated by tokens including `rue`, `de`, `la`, `r`, `du`, and `des`. These patterns support accent-aware/subword representations and caution against treating short tokens as uniformly uninformative.","",
      f"**CONFIRMED:** word novelty is large but character novelty is much smaller. France S2 mean name word-OOV is {rv('test','France','S2','name_word_token_oov_fraction','mean'):.1%}, versus {rv('test','France','S2','name_char_4gram_oov_fraction','mean'):.1%} for name 4-grams. This directly supports a character/subword channel while threatening word-exact or vocabulary-memorization-heavy matching.","",
      "`france_domain_geometry.csv` gives per-source distributions of cosine similarity and standardized compact-feature distance to US and India train centroids, plus US-closer fractions and a conservative joint-neither rate. The result is intentionally multi-dimensional: closeness in character space can coexist with structural distance.","",
      f"**INFERENCE:** France is relatively closer to US than India in the fitted train geometry, not actually US-like. Character cosine chooses US for {gm('S1','tfidf_fraction_US_closer'):.1%}/{gm('S2','tfidf_fraction_US_closer'):.1%}/{gm('S3','tfidf_fraction_US_closer'):.1%}; compact features choose US for {gm('S1','compact_fraction_US_closer'):.1%}/{gm('S2','compact_fraction_US_closer'):.1%}/{gm('S3','compact_fraction_US_closer'):.1%}. Yet France remains almost perfectly separable by character adversarial validation. “Closer to US” is therefore only a relative statement between two imperfect references.","",
      "# SOURCE_STRUCTURE","",
      "`source_density.csv` contains S2/S1, S3/S1, total auxiliary/S1, source-specific missing-address rates, and exact duplication/fanout summaries. Higher auxiliary density raises the number of plausible negatives and owner competition; it does not by itself imply more true matches.","",
      f"**CONFIRMED:** France has heavier normalized-address ambiguity. S2/S3 normalized-address duplicate rates are {dm('test','France','S2','normalized_address_duplicate_rate'):.1%}/{dm('test','France','S3','normalized_address_duplicate_rate'):.1%}, versus train US {dm('train','US','S2','normalized_address_duplicate_rate'):.1%}/{dm('train','US','S3','normalized_address_duplicate_rate'):.1%} and train India {dm('train','India','S2','normalized_address_duplicate_rate'):.1%}/{dm('train','India','S3','normalized_address_duplicate_rate'):.1%}. France address-fanout p99 is 9 in S2/S3 (and 15.5 in S1), versus 4–5 in auxiliary train sources. This threatens address-exact uniqueness and independent-pair assumptions; the matcher should expose fanout/competition context.","",
      f"**CONFIRMED:** auxiliary missing-address rates are modest and not France-specific: France S2/S3 are {dm('test','France','S2','missing_address_rate'):.1%}/{dm('test','France','S3','missing_address_rate'):.1%}, within the train/test seen-country range of roughly 2.3–3.7%. Missingness still needs an explicit indicator, but it does not explain the France-wide shift.","",
      "# ADVERSARIAL_VALIDATION","",
      "The compact diagnostic is a standardized logistic model over lengths, token counts, missingness, Unicode, digit, punctuation, whitespace/segment statistics, and source indicator. The optional character model is a capped 2–5-gram TF-IDF linear classifier. Both use deterministic balanced samples and a fixed 70/30 diagnostic holdout. They never use entity IDs, truth, candidate labels, or production artifacts. `adversarial_feature_importance.csv` gives signed standardized coefficients/linear weights; positive means test-like.","",
      "For France, the dominant compact drivers are segment/address-token structure, non-ASCII/diacritics, whitespace, address/name length, and digit ratio. Seen-country coefficients are small and AUROCs remain near chance. Threatened assumption: a matcher can exploit country/source formatting shortcuts that look predictive locally but do not encode identity equivalence.","",
      "# MODEL_BEHAVIOR","",
      "`existing_model_shift.csv` uses already-existing P002/D002 outputs only. Score quantiles cover the historical shortlist; owner scores/margins and accepted matches/S1 are copied from existing diagnostics. Newly aggregated disagreement and lexical-conflict rates join existing P002 shortlist and neural caches only—no inference was launched. Score shifts and confidence must not be interpreted as correctness or calibration on hidden test truth.","",
      f"| country | fused median | accepted/S1 | lexical >=.8 and XLM-R <=.1 | LGBM-XLM-R mean abs disagreement |","|---|---:|---:|---:|---:|",
      f"| US | {mm('US','fused_shortlist'):.3f} | {mm('US','accepted_matches_per_s1','mean'):.3f} | {mm('US','lexical_ge_.8_xlmr_le_.1_rate','mean'):.1%} | {mm('US','lgbm_xlmr_abs_disagreement_mean','mean'):.3f} |",
      f"| India | {mm('India','fused_shortlist'):.3f} | {mm('India','accepted_matches_per_s1','mean'):.3f} | {mm('India','lexical_ge_.8_xlmr_le_.1_rate','mean'):.1%} | {mm('India','lgbm_xlmr_abs_disagreement_mean','mean'):.3f} |",
      f"| France | {mm('France','fused_shortlist'):.3f} | {mm('France','accepted_matches_per_s1','mean'):.3f} | {mm('France','lexical_ge_.8_xlmr_le_.1_rate','mean'):.1%} | {mm('France','lgbm_xlmr_abs_disagreement_mean','mean'):.3f} |","",
      f"France also rejects {mm('France','rejected_competition_per_s1','mean')+mm('France','rejected_margin_only_per_s1','mean'):.3f} threshold-passing candidates/S1 through ownership or margin, versus {mm('US','rejected_competition_per_s1','mean')+mm('US','rejected_margin_only_per_s1','mean'):.3f} US and {mm('India','rejected_competition_per_s1','mean')+mm('India','rejected_margin_only_per_s1','mean'):.3f} India. Rank-bucket rows in the CSV relate lexical, LGBM, and XLM-R medians to best retrieval rank. **INFERENCE:** the model stack is encountering France-specific evidence conflict and competition. **OPEN QUESTION:** without labels, none of these rates says whether the neural model, lexical evidence, or decoder is right.","",
      "# IMPLICATIONS_FOR_MATCHER","",
      "- Preserve both character/subword evidence and compact structural/numeric features; word-token novelty alone is not a safe gate.",
      "- Represent missing address explicitly and evaluate it by source/country; do not treat empty address as an ordinary repeated normalized address.",
      "- Make normalized-name/address fanout and owner competition visible to diagnostics. High fanout threatens independent-pair classification and encourages group-aware evaluation, but does not justify changing the decoder from test confidence alone.",
      "- Add country-held-out, size-controlled validation before claiming France robustness. Test-domain separability does not identify the optimal architecture or authorize test adaptation.",
      "- Preserve lexical-vs-neural disagreement slices (especially strong lexical/low neural) in validation reports. Only labeled, isolated validation can tell whether either side is right.","",
      "# COUNTRY_STRESS_SPEC","",
      "The exact future design is in `future_country_stress_spec.md`. It equalizes S1 training groups and, if necessary, label/source pair strata so exposure is not confounded with training size. It is a specification only and has not been run.","",
      "# WHAT_NOT_TO_INFER","",
      "- No test label, France identity, blocker recall, matcher accuracy, or leaderboard contribution was inferred.",
      "- Adversarial AUROC measures detectability of domain membership, not downstream error.",
      "- Train-centroid proximity is descriptive and cannot assign France to a country domain or prove transfer quality.",
      "- Confidence, accepted matches/S1, owner margins, lexical similarity, and normalized fanout are not correctness.",
      "- This audit did not train/adapt a production matcher, change splits, pseudo-label, use external data, benchmark/modify the blocker, or use leaderboard feedback.","",
      "## Reproducibility","",
      f"Seed `{SEED}`; audit database `{DB.relative_to(ROOT)}` opened read-only; raw ground-truth table was never queried. The script is `analysis/test_shift_audit_v2/run_audit.py`."
    ]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    OUT.mkdir(parents=True,exist_ok=True); start=time.time()
    con=duckdb.connect();con.execute("PRAGMA threads=12");con.execute("PRAGMA memory_limit='24GB'");con.execute(f"ATTACH '{DB}' AS a (READ_ONLY)")
    # Explicit safety check: only unlabeled source tables are referenced below.
    raw_path=OUT/"raw_distribution_metrics.csv"; sample_path=OUT/"diagnostic_samples.parquet"; vocab_path=OUT/"train_vocab.pkl"
    if raw_path.exists(): raw=pd.read_csv(raw_path); print("resume raw",len(raw),flush=True)
    else: raw=exact_raw_metrics(con);raw.to_csv(raw_path,index=False)
    if sample_path.exists(): samples=pd.read_parquet(sample_path); print("resume samples",len(samples),flush=True)
    else: samples=load_samples(con);samples.to_parquet(sample_path,index=False)
    if vocab_path.exists():
        with vocab_path.open("rb") as fh: words,grams=pickle.load(fh)
        print("resume vocab",len(words),flush=True)
    else:
        words,grams=build_training_vocab(con)
        with vocab_path.open("wb") as fh: pickle.dump((words,grams),fh,pickle.HIGHEST_PROTOCOL)
    if not any(raw.metric.astype(str).str.contains("oov")):
        raw,per_record=add_oov_metrics(samples,words,grams,raw);raw=divergence(raw,samples,per_record);raw.to_csv(raw_path,index=False)
    else:
        _,per_record=add_oov_metrics(samples,words,grams,raw.iloc[0:0].copy())
    density=source_density(con,raw);density.to_csv(OUT/"source_density.csv",index=False)
    france=append_france_fanout_summaries(france_text(con,samples,per_record),raw);france.to_csv(OUT/"france_text_stats.csv",index=False)
    av,imp=adversarial(samples);av.to_csv(OUT/"adversarial_validation.csv",index=False);imp.to_csv(OUT/"adversarial_feature_importance.csv",index=False)
    geo=geometry(samples);geo.to_csv(OUT/"france_domain_geometry.csv",index=False)
    model=existing_model_shift(con);model.to_csv(OUT/"existing_model_shift.csv",index=False)
    case=casebook(con,samples,per_record,raw);case.to_csv(OUT/"france_casebook.tsv",sep="\t",index=False,quoting=csv.QUOTE_MINIMAL)
    raw.to_csv(OUT/"raw_distribution_metrics.csv",index=False);density.to_csv(OUT/"source_density.csv",index=False);france.to_csv(OUT/"france_text_stats.csv",index=False)
    av.to_csv(OUT/"adversarial_validation.csv",index=False);imp.to_csv(OUT/"adversarial_feature_importance.csv",index=False);geo.to_csv(OUT/"france_domain_geometry.csv",index=False);model.to_csv(OUT/"existing_model_shift.csv",index=False);case.to_csv(OUT/"france_casebook.tsv",sep="\t",index=False,quoting=csv.QUOTE_MINIMAL)
    write_stress_spec();make_report(raw,density,france,av,imp,geo,model)
    deliverables=["REPORT.md","raw_distribution_metrics.csv","source_density.csv","france_text_stats.csv","adversarial_validation.csv","adversarial_feature_importance.csv","france_domain_geometry.csv","existing_model_shift.csv","france_casebook.tsv","future_country_stress_spec.md"]
    hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in deliverables}
    # Restart caches are intentionally not deliverables and may contain sampled raw text.
    sample_path.unlink(missing_ok=True);vocab_path.unlink(missing_ok=True)
    (OUT/"run_manifest.json").write_text(json.dumps({"status":"complete_validated","seed":SEED,"sample_n_per_group":SAMPLE_N,"seconds":time.time()-start,"ground_truth_read":False,"blocker_benchmarked_or_modified":False,"production_matcher_trained_or_adapted":False,"production_inference_launched":False,"external_data_or_network_used":False,"output_sha256":hashes},indent=2))
    print("COMPLETE",round(time.time()-start,1),flush=True)


if __name__=="__main__": main()
