#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

rows=[]
text=subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version","--format=csv,noheader,nounits"],text=True,capture_output=True,check=True).stdout
for line in text.strip().splitlines():
    i,u,n,total,used,free,util,driver=[x.strip() for x in line.split(",")]
    rows.append({"index":int(i),"uuid":u,"name":n,"memory_total_mib":int(total),"memory_used_mib":int(used),"memory_free_mib":int(free),"utilization":int(util),"driver":driver})
ada=next((x for x in rows if "6000 Ada" in x["name"] and x["memory_used_mib"]<1024),None)
a6000=next((x for x in rows if x["name"]=="NVIDIA RTX A6000" and x["memory_used_mib"]<1024),None)
chosen=ada or a6000
if chosen is None:raise SystemExit("Neither preferred GPU is idle; refusing to interfere with another process")
out={"inventory":rows,"chosen":chosen,"policy":"idle Ada preferred, otherwise idle A6000"}
p=Path("analysis/production/p001_e005b/diagnostics/gpu_selection.json");p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+"\n")
print(chosen["uuid"])
