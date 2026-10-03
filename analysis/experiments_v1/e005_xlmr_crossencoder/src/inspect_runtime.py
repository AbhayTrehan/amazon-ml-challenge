#!/usr/bin/env python3
import json, os, platform, subprocess

def cmd(args):
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()

info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "ram": cmd(["bash", "-lc", "free -h"]),
    "gpu": cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    "cuda_smi": cmd(["nvidia-smi"]),
    "cuda_compiler": cmd(["bash", "-lc", "nvcc --version | tail -n 1"]),
}
try:
    import torch
    info.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                bf16_supported=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
except Exception as e:
    info["torch_error"] = repr(e)
try:
    import transformers
    info["transformers_version"] = transformers.__version__
except Exception as e:
    info["transformers_error"] = repr(e)
print(json.dumps(info, indent=2))
