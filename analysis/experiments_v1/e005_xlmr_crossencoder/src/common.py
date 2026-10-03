from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SEED = 20260925
MODEL_ID = "FacebookAI/xlm-roberta-base"
MISSING_ADDRESS = "[missing]"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def serialize(name, address) -> str:
    addr = MISSING_ADDRESS if pd.isna(address) or str(address).strip() == "" else str(address)
    safe_name = "" if pd.isna(name) else str(name)
    return f"name: {safe_name}\naddress: {addr}"


def shortlist(d: pd.DataFrame) -> pd.DataFrame:
    use = d.lightgbm_rank_within_s1.le(5) | d.lightgbm_score.ge(0.05)
    return d.loc[use].copy().reset_index(drop=True)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.a = [serialize(n, a) for n, a in zip(frame.s1_business_name, frame.s1_business_address)]
        self.b = [serialize(n, a) for n, a in zip(frame.candidate_business_name, frame.candidate_business_address)]
        self.labels = frame.label.to_numpy(dtype=np.int64) if "label" in frame else None

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], None if self.labels is None else int(self.labels[i])


class PairCollator:
    def __init__(self, tokenizer, max_length: int, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, rows):
        a, b, labels = zip(*rows)
        batch = self.tokenizer(list(a), list(b), truncation=True, max_length=self.max_length,
                               padding=True, pad_to_multiple_of=self.pad_to_multiple_of,
                               return_tensors="pt")
        if labels[0] is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


class TokenizedPairDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = np.fromiter((len(x) for x in input_ids), dtype=np.int16, count=len(input_ids))
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): return {"input_ids": self.input_ids[i], "attention_mask": [1]*len(self.input_ids[i]), "labels": int(self.labels[i])}


class TokenizedCollator:
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def __call__(self, rows):
        labels=torch.tensor([r.pop("labels") for r in rows],dtype=torch.long)
        batch=self.tokenizer.pad(rows,padding=True,pad_to_multiple_of=8,return_tensors="pt")
        batch["labels"]=labels;return batch


class LengthBucketBatchSampler(torch.utils.data.Sampler):
    """Deterministic shuffled batches with low padding waste."""
    def __init__(self, lengths, batch_size, seed, bucket_batches=50):
        self.lengths=np.asarray(lengths);self.batch_size=batch_size;self.seed=seed;self.bucket_batches=bucket_batches;self.epoch=0
    def set_epoch(self, epoch): self.epoch=int(epoch)
    def __len__(self): return int(np.ceil(len(self.lengths)/self.batch_size))
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);order=rng.permutation(len(self.lengths));batches=[]
        width=self.batch_size*self.bucket_batches
        for start in range(0,len(order),width):
            pool=order[start:start+width];pool=pool[np.argsort(self.lengths[pool],kind="stable")]
            batches.extend(pool[i:i+self.batch_size].tolist() for i in range(0,len(pool),self.batch_size))
        rng.shuffle(batches);yield from batches


def pretokenize(frame, tokenizer, max_length, cache_path=None):
    cache_path=Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        saved=torch.load(cache_path,map_location="cpu",weights_only=False)
        if saved.get("max_length")==max_length and len(saved["input_ids"])==len(frame): return TokenizedPairDataset(saved["input_ids"],frame.label)
    ids=[]
    for start in range(0,len(frame),2048):
        x=frame.iloc[start:start+2048]
        a=[serialize(n,z) for n,z in zip(x.s1_business_name,x.s1_business_address)]
        b=[serialize(n,z) for n,z in zip(x.candidate_business_name,x.candidate_business_address)]
        ids.extend(tokenizer(a,b,truncation=True,max_length=max_length,padding=False)["input_ids"])
    if cache_path:
        cache_path.parent.mkdir(parents=True,exist_ok=True);torch.save({"max_length":max_length,"input_ids":ids},cache_path)
    return TokenizedPairDataset(ids,frame.label)


def resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()
