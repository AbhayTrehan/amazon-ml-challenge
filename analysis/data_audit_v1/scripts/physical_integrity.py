#!/usr/bin/env python3
"""Byte-level checks that do not modify source files."""
import csv,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'analysis/data_audit_v1/artifacts'; files=sorted((ROOT/'student_resource/dataset').glob('*/*.tsv'));out=[]
for p in files:
 h=hashlib.sha256(); lines=blank=badtab=quotes=controls=crlf=0; exp=1 if 'ground_truth' in p.name else 3
 with p.open('rb')as f:
  for line in f:
   h.update(line);lines+=1;crlf+=line.endswith(b'\r\n');body=line.rstrip(b'\r\n');blank+=not body;badtab+=body.count(b'\t')!=exp;quotes+=b'"'in body;controls+=sum(c<32 and c not in(9,)for c in body)
 out.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'physical_lines':lines,'data_rows':lines-1,'sha256':h.hexdigest(),'utf8_bom':p.read_bytes()[:3]==b'\xef\xbb\xbf','newline_style':'CRLF'if crlf==lines else('mixed'if crlf else'LF'),'blank_lines':blank,'unexpected_tab_count_lines':badtab,'lines_containing_quote':quotes,'embedded_control_bytes':controls})
 print(p.name,lines)
with (OUT/'physical_integrity.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
