#!/usr/bin/env python3
import shutil
import pandas as pd
from transformers import AutoTokenizer
from common import EXP, MODEL_ID, MODEL_REVISION
from neural import encode_frame

tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
val=pd.read_parquet(EXP/"cache/shortlist/validation.parquet")
master=EXP/"runtime/validation_tokens.pt"
encode_frame(val,tok,master)
target=EXP/"cache/neural/validation_selected.tokens.pt"
target.parent.mkdir(parents=True,exist_ok=True)
if target.exists():target.unlink()
target.symlink_to(master.resolve())
print(f"VALIDATION_TOKENS_READY rows={len(val)} cache={master}",flush=True)
