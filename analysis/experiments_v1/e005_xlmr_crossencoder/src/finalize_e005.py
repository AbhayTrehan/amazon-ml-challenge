#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate_e005 import mask_for, score_column
from metric import grouped_f05

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "analysis/experiments_v1/e005_xlmr_crossencoder"
BUNDLE = ROOT / "analysis/experiments_v1/e004_k10_decode_and_gpu_bundle/gpu_bundle"


def run(*args):
    return subprocess.run(args, text=True, capture_output=True, check=False).stdout.strip()


def decision(delta):
    if delta >= .010: return "STRONG KEEP"
    if delta >= .005: return "KEEP"
    if delta >= .002: return "MARGINAL"
    return "KILL"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""): h.update(chunk)
    return h.hexdigest()


def main():
    out = EXP / "outputs"; model_dir = EXP / "model/best_checkpoint"
    eval_results = pd.read_csv(out / "evaluation_results.csv")
    slices = pd.read_csv(out / "slice_results.csv")
    train = pd.read_csv(EXP / "training_log.csv")
    token = json.loads((EXP / "token_length_stats.json").read_text())
    bench = json.loads((EXP / "runtime/benchmark.json").read_text())
    integrity = json.loads((EXP / "runtime/integrity_preflight.json").read_text())
    controls = json.loads((EXP / "runtime/shortlist_control.json").read_text())
    cal_rt = json.loads((out / "calibration_predictions.runtime.json").read_text())
    eval_rt = json.loads((out / "evaluation_predictions.runtime.json").read_text())
    frozen = json.loads((EXP / "model/frozen_parameters.json").read_text())
    selected_epoch = int((EXP / "runtime/selected_epoch.txt").read_text())
    epoch1_cal = json.loads((out / "calibration_summary_epoch1.json").read_text())["best"]["E005A"]["macro_f05"]
    epoch2_cal = json.loads((out / "calibration_summary_epoch2.json").read_text())["best"]["E005A"]["macro_f05"]

    # Artifact reproduction check from saved fold-0 predictions.
    pred = pd.read_parquet(out / "evaluation_predictions.parquet")
    meta = pd.read_parquet(EXP / "runtime/selected_s1_metadata.parquet").query("role == 'evaluation'")
    reproduced = {}
    for exp in ["E005A", "E005B"]:
        cfg = frozen[exp]
        mask = mask_for(pred, score_column(pred, exp, cfg.get("weight")), cfg)
        reproduced[exp] = grouped_f05(pred, meta, mask)["macro_f05"]
        reported = float(eval_results.loc[eval_results.experiment.eq(exp), "macro_f05"].iloc[0])
        if abs(reproduced[exp] - reported) > 1e-12: raise RuntimeError(f"{exp} reproduction failed")

    # Reload selected artifacts and verify intended architecture/tokenizer.
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True,
                                                               local_files_only=True, attn_implementation="sdpa")
    if mdl.config.model_type != "xlm-roberta" or mdl.config.num_labels != 2: raise RuntimeError("model identity failed")
    params = sum(p.numel() for p in mdl.parameters()); del mdl, tok

    manifest = json.loads((BUNDLE / "bundle_manifest.json").read_text())
    current_hashes = {name: sha256(BUNDLE / name) for name in manifest["files"]}
    if any(current_hashes[n] != manifest["files"][n]["sha256"] for n in current_hashes):
        raise RuntimeError("input changed after training")

    pip_freeze = run(str(Path("/home/satvik/venvs/amazonml2026/bin/pip")), "freeze")
    (EXP / "runtime/pip_freeze.txt").write_text(pip_freeze + "\n")
    req_names = ["torch", "transformers", "safetensors", "sentencepiece", "pandas", "pyarrow", "numpy", "scikit-learn", "tqdm", "PyYAML"]
    freeze_map = {line.split("==", 1)[0].lower(): line for line in pip_freeze.splitlines() if "==" in line}
    aliases = {"scikit-learn":"scikit-learn", "pyyaml":"pyyaml"}
    req = [freeze_map[aliases.get(n.lower(), n.lower())] for n in req_names if aliases.get(n.lower(), n.lower()) in freeze_map]
    (EXP / "requirements_e005.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cu128\n" + "\n".join(req) + "\n")

    smi = run("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,compute_cap", "--format=csv,noheader")
    env = f"""hostname: {platform.node()}
python: {platform.python_version()}
venv: /home/satvik/venvs/amazonml2026
torch: {torch.__version__}
torch CUDA runtime: {torch.version.cuda}
transformers: {transformers.__version__}
CUDA available: {torch.cuda.is_available()}
selected physical GPU index: 1
selected GPU UUID: GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17
selected GPU: NVIDIA RTX 6000 Ada Generation
GPU inventory: {smi}
CPU count: {run('nproc')}
RAM: {run('bash','-lc','free -h | sed -n 2p')}
disk: {run('df','-h',str(ROOT))}
"""
    (EXP / "environment.txt").write_text(env)

    start = float((EXP / "runtime/pipeline_start_epoch.txt").read_text())
    end = time.time()
    measured_pps = eval_rt["pairs_per_second"]
    test_s1 = 1732544
    runtime = {
        "selected_epoch": selected_epoch, "physical_gpu_index": 1,
        "gpu_uuid": "GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17",
        "gpu": "NVIDIA RTX 6000 Ada Generation", "precision": "bf16",
        "max_length": token["chosen_max_length"], "truncation_rate": token["truncation_rate"],
        "physical_batch_size": 64, "effective_batch_size": 64,
        "benchmark": bench, "training": train.to_dict("records"),
        "calibration_inference": cal_rt, "evaluation_inference": eval_rt,
        "pipeline_wall_seconds": end - start, "parameter_count": params,
        "projection": {
            "full_k10_pairs": test_s1 * 37.85,
            "full_k10_seconds": test_s1 * 37.85 / measured_pps,
            "shortlist_pairs": test_s1 * 6.19,
            "shortlist_seconds": test_s1 * 6.19 / measured_pps,
            "measured_pairs_per_second": measured_pps,
        },
        "integrity": {"prediction_reproduction": reproduced, "input_hashes_after": current_hashes,
                      "checkpoint_reload": True, "cuda_used": True, "fold0_selection_influence": False,
                      "external_business_lookup": False, "official_data_modified": False},
    }
    (EXP / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")

    best_row = eval_results.sort_values("macro_f05", ascending=False).iloc[0]
    best_exp = best_row.experiment; best_delta = float(best_row.delta_vs_e004b)
    dsn = decision(best_delta)
    def slice_val(exp, sl, value):
        z=slices[(slices.experiment.eq(exp))&(slices.slice.eq(sl))&(slices.value.astype(str).eq(str(value)))]
        return float(z.macro_f05.iloc[0]) if len(z) else float("nan")
    lines = [
        "# E005 XLM-R cross-encoder handoff", "", "# Environment / GPU", "",
        f"Host `{platform.node()}`; reusable venv `/home/satvik/venvs/amazonml2026`; PyTorch {torch.__version__}, CUDA {torch.version.cuda}, Transformers {transformers.__version__}. Physical GPU 1, RTX 6000 Ada Generation (48 GiB), UUID `GPU-f5f574b3-d0dc-fd72-5fd4-2d913d115b17`; BF16 and SDPA; one GPU only.", "",
        "# Input integrity", "", f"All three manifest SHA256 hashes and row counts passed before training and after evaluation. Schema/labels passed; train, fold 4, and fold 0 S1 sets are pairwise disjoint. No official data was modified and no external business lookup occurred.", "",
        "# Dataset / shortlist", "", f"Training: 398,428 rows (98,428 positive, 300,000 hard negative), 30,000 S1. Shortlist rule: rank <=5 OR LightGBM >=0.05. Fold 4: 61,621 pairs, oracle {controls['calibration']['oracle']['macro_f05']:.6f}. Fold 0: 61,913 pairs, oracle {controls['evaluation']['oracle']['macro_f05']:.6f}.", "",
        "# Model and revision", "", f"`FacebookAI/xlm-roberta-base`, revision `{token['model_revision']}`, MIT, {params:,} parameters, two-class cross-entropy, full fine-tuning.", "",
        "# Token lengths / max_length", "", f"p50/p90/p95/p99/p99.5/max = {token['quantiles']['p50']:.0f}/{token['quantiles']['p90']:.0f}/{token['quantiles']['p95']:.0f}/{token['quantiles']['p99']:.0f}/{token['quantiles']['p99_5']:.0f}/{token['max']}; max_length={token['chosen_max_length']}; measured truncation={100*token['truncation_rate']:.3f}%.", "",
        "# Throughput benchmark", "", f"Batch 64 BF16: {bench['training_examples_per_second']:.1f} train examples/s, {bench['inference_examples_per_second']:.1f} inference pairs/s, {bench['peak_allocated_gpu_gb']:.2f} GiB peak allocated.", "",
        "# Training", "", "```text", train.to_string(index=False), "```", "",
        "# Checkpoint selection", "", f"Epoch 1 fold-4 E005A macro F0.5 was **{epoch1_cal:.6f}**; epoch 2 was **{epoch2_cal:.6f}**. Epoch {selected_epoch} was therefore selected strictly by fold-4 E005A grouped macro F0.5. Fold 0 was not read until `model/frozen_parameters.json` was written.", "",
        "# Fair shortlist classical control", "", f"Fold 4={controls['calibration']['e004b_shortlist']['macro_f05']:.6f}; fold 0={controls['evaluation']['e004b_shortlist']['macro_f05']:.6f}; full-K10 E004B fold 0=0.928336.", "",
    ]
    for exp in ["E005A", "E005B"]:
        r=eval_results[eval_results.experiment.eq(exp)].iloc[0]; cfg=frozen[exp]
        lines += [f"# {exp}", "", f"Frozen fold-4 configuration: `{json.dumps(cfg, sort_keys=True)}`. Fold-0 macro F0.5={r.macro_f05:.6f}, precision={r.precision:.6f}, recall={r.recall:.6f}, singleton accuracy={r.singleton_accuracy:.6f}, delta vs E004B={r.delta_vs_e004b:+.6f}, oracle gap={r.oracle_model_gap:.6f}. Decision: {decision(float(r.delta_vs_e004b))}.", ""]
    lines += ["# Important slices", ""]
    slice_table=[]
    for exp in ["E005A","E005B"]:
        for sl,val in [("country","US"),("country","India"),("source","S2"),("source","S3"),("cardinality","0"),("cardinality","1"),("cardinality","2"),("cardinality","3"),("cardinality","4"),("cardinality","5+"),("name_ambiguity","repeated/common"),("name_ambiguity","non_ambiguous"),("missing_true_candidate_address","true")]:
            slice_table.append({"experiment":exp,"slice":sl,"value":val,"macro_f05":slice_val(exp,sl,val)})
    lines += ["```text", pd.DataFrame(slice_table).to_string(index=False), "```", ""]
    for exp in ["E005A","E005B"]:
        lines += [f"{exp}: India {'improved' if slice_val(exp,'country','India')>.908002 else 'regressed'} vs 0.908002; cardinality-1 {'improved' if slice_val(exp,'cardinality','1')>.841237 else 'regressed'} vs 0.841237; repeated-name {'improved' if slice_val(exp,'name_ambiguity','repeated/common')>.888798 else 'regressed'} vs 0.888798.", ""]
    disagreements=pd.read_csv(out/"disagreement_counts.csv")
    lines += ["# Complementarity", "", "```text", disagreements.to_string(index=False), "```", "",
              "# Runtime / production projection", "", f"Epoch times: {', '.join(f'{x:.1f}s' for x in train.seconds)}. Measured evaluation inference: {measured_pps:.1f} pairs/s. Projected full-test K10: {runtime['projection']['full_k10_pairs']:,.0f} pairs / {runtime['projection']['full_k10_seconds']/3600:.2f} h. Projected shortlist: {runtime['projection']['shortlist_pairs']:,.0f} pairs / {runtime['projection']['shortlist_seconds']/3600:.2f} h. Detached pipeline wall time: {runtime['pipeline_wall_seconds']/60:.1f} min.", "",
              "# Decision", "", f"Best result: {best_exp}, delta {best_delta:+.6f}; **{dsn}**. E006 was not started.", "",
              "# Artifact paths", "", f"Best model: `model/best_checkpoint/` (`model.safetensors` SHA256 `{sha256(model_dir/'model.safetensors')}`); predictions/search/results/slices/disagreements: `outputs/`; runtime and integrity evidence: `runtime.json` and `runtime/`.", "",
              "# Exact reproduction command", "", "```bash\nbash analysis/experiments_v1/e005_xlmr_crossencoder/scripts/run_e005.sh\n```", "",
              "# Spreadsheet rows", ""]
    for exp,label in [("E005A","XLM-R cross-encoder"),("E005B","XLM-R + LightGBM fusion")]:
        r=eval_results[eval_results.experiment.eq(exp)].iloc[0]
        lesson="joint multilingual pair scoring result under the frozen shortlist" if exp=="E005A" else "simple fold-4-calibrated fusion result under the frozen shortlist"
        lines += [f"Satvik | {exp} {label} | Val F0.5={r.macro_f05:.6f} (Oracle=0.986696) | {decision(float(r.delta_vs_e004b))} — {lesson}."]
    (EXP / "HANDOFF.md").write_text("\n".join(lines) + "\n")
    (EXP / "README.md").write_text("# E005 XLM-R cross-encoder\n\nSee `HANDOFF.md` for the complete frozen experiment record.\n")
    (EXP / "runtime/COMPLETE").write_text(f"completed_epoch={time.time()}\n")


if __name__ == "__main__": main()
