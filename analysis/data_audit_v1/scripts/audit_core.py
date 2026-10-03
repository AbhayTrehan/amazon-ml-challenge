#!/usr/bin/env python3
"""Full streaming physical/schema/text audit and compact SQLite train index."""
from __future__ import annotations
import csv, json, os, sqlite3, sys, time, unicodedata, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from common import *

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "student_resource/dataset"
OUT = ROOT / "analysis/data_audit_v1/artifacts"
DB = OUT / "audit.sqlite"
SOURCE_FILES = [*(DATA/"train").glob("*.tsv"), *(DATA/"test").glob("*.tsv")]


class FieldStats:
    def __init__(self):
        self.n=self.null=self.empty=self.ws=self.sentinel=self.nonascii=0
        self.length=Counter(); self.tok=Counter(); self.digit=Counter(); self.script=Counter()
    def add(self, s):
        self.n += 1
        if s is None: self.null += 1; return
        if s == "": self.empty += 1
        if s and not s.strip(): self.ws += 1
        if s.strip().casefold() in SENTINELS: self.sentinel += 1
        self.length[len(s)] += 1; self.tok[len(tokens(s))] += 1
        self.digit[sum(c.isdigit() for c in s)] += 1
        if any(ord(c)>127 for c in s): self.nonascii += 1
        self.script[script_bucket(s)] += 1
    def report(self):
        q=lambda h: {**quantiles_from_counter(h), "mean": sum(k*v for k,v in h.items())/sum(h.values()) if h else None}
        return {"n":self.n,"null":self.null,"empty":self.empty,"whitespace_only":self.ws,"sentinel":self.sentinel,
                "nonascii":self.nonascii,"nonascii_rate":self.nonascii/self.n if self.n else 0,
                "length":q(self.length),"tokens":q(self.tok),"digits":q(self.digit),"scripts":dict(self.script)}


def physical(path):
    raw = path.read_bytes()[:4]
    bom = raw.startswith(b"\xef\xbb\xbf")
    crlf=lf=cr=blank=bad_tabs=controls=quote_lines=0
    expected = 1 if "ground_truth" in path.name else 3
    sha=hashlib.sha256()
    with path.open("rb") as f:
        for line in f:
            sha.update(line); lf += line.endswith(b"\n"); crlf += line.endswith(b"\r\n")
            if line.rstrip(b"\r\n") == b"": blank += 1
            if line.rstrip(b"\r\n").count(b"\t") != expected: bad_tabs += 1
            quote_lines += b'"' in line
            controls += sum(1 for b in line.rstrip(b"\r\n\t") if b<32 and b not in (9,))
    return {"bytes":path.stat().st_size,"physical_lines":lf,"bom":bom,"newline":"CRLF" if crlf==lf else ("mixed" if crlf else "LF"),
            "blank_lines":blank,"unexpected_tab_count_lines":bad_tabs,"lines_with_quote":quote_lines,"embedded_controls":controls,"sha256":sha.hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True); start=time.time()
    conn=sqlite3.connect(DB); conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("DROP TABLE IF EXISTS records")
    conn.execute("CREATE TABLE records(split TEXT, source INTEGER, id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT, rownum INTEGER, nname TEXT, naddr TEXT, sname TEXT)")
    conn.execute("DROP TABLE IF EXISTS gt")
    conn.execute("CREATE TABLE gt(s1 TEXT, target TEXT, ord INTEGER, PRIMARY KEY(s1,target))")
    reports={}; inventory=[]
    global_ids=set(); cross_file_dups=Counter()
    for path in sorted(SOURCE_FILES):
        split=path.parent.name
        is_gt="ground_truth" in path.name
        src=0 if is_gt else int(path.stem[-1])
        phy=physical(path); stats={"name":FieldStats(),"address":FieldStats()}; countries=Counter()
        rows=bad=dup_rows=dup_ids=missing_id=bad_prefix=both_missing=order_increasing=0
        prior_num=None; seen=set(); fullrow=set(); batch=[]; gtbatch=[]; header=[]
        with path.open(encoding="utf-8", newline="") as f:
            reader=csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header=next(reader)
            for rownum,row in enumerate(reader,1):
                rows += 1
                if len(row) != (2 if is_gt else 4): bad += 1; continue
                if tuple(row) in fullrow: dup_rows += 1
                else: fullrow.add(tuple(row))
                eid=row[0]
                if not eid: missing_id += 1
                if eid in seen: dup_ids += 1
                seen.add(eid)
                if eid in global_ids: cross_file_dups[eid]+=1
                global_ids.add(eid)
                expected="S1-" if (is_gt or src==1) else f"S{src}-"
                if not eid.startswith(expected) or numeric_suffix(eid) is None: bad_prefix += 1
                num=numeric_suffix(eid)
                if prior_num is not None and num is not None and num>prior_num: order_increasing += 1
                prior_num=num
                if is_gt:
                    vals=[] if not row[1] else row[1].split(",")
                    for j,target in enumerate(vals): gtbatch.append((eid,target,j))
                    if len(gtbatch)>=100000: conn.executemany("INSERT OR IGNORE INTO gt VALUES(?,?,?)",gtbatch); gtbatch=[]
                else:
                    name,address,country=row[1:]; countries[country]+=1
                    stats["name"].add(name); stats["address"].add(address)
                    both_missing += (not name and not address)
                    batch.append((split,src,eid,name,address,country,rownum,punct_norm(name),punct_norm(address),suffix_norm(name)))
                    if len(batch)>=50000:
                        conn.executemany("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?,?,?)",batch); conn.commit(); batch=[]
        if batch: conn.executemany("INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?,?,?)",batch)
        if gtbatch: conn.executemany("INSERT OR IGNORE INTO gt VALUES(?,?,?)",gtbatch)
        conn.commit()
        rep={**phy,"parsed_rows":rows,"columns":header,"column_count":len(header),"malformed_rows":bad,
             "duplicate_rows":dup_rows,"duplicate_entity_ids":dup_ids,"missing_entity_id":missing_id,"bad_id_prefix_or_numeric":bad_prefix,
             "adjacent_numeric_increasing_fraction":order_increasing/max(1,rows-1)}
        if not is_gt: rep.update({"source":src,"split":split,"countries":dict(countries),"both_name_address_empty":both_missing,"name":stats['name'].report(),"address":stats['address'].report()})
        reports[str(path.relative_to(ROOT))]=rep
        inventory.append({"path":str(path.relative_to(ROOT)),"bytes":phy["bytes"],"rows":rows,"purpose":"training labels" if is_gt else f"{split} source {src} records"})
        print(path.name, rows, round(time.time()-start,1), flush=True)
    conn.executescript("CREATE INDEX idx_records_lookup ON records(split,id); CREATE INDEX idx_exact_name ON records(split,source,country,nname); CREATE INDEX idx_exact_addr ON records(split,source,country,naddr); CREATE INDEX idx_gt_s1 ON gt(s1);")
    conn.commit()
    reports["cross_file_duplicate_ids"]={"count":len(cross_file_dups),"examples":list(cross_file_dups)[:20]}
    reports["runtime_seconds"]=time.time()-start
    (OUT/"physical_schema_text.json").write_text(json.dumps(reports,indent=2,ensure_ascii=False))
    with (OUT/"dataset_inventory.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=inventory[0].keys()); w.writeheader(); w.writerows(inventory)
    print("DONE", time.time()-start)

if __name__=="__main__": main()
