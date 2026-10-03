"""Preset remote stages invoked by Colab CLI; ordinary subprocess scripts, no notebook."""
import json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT=Path("/content/e005"); SRC=ROOT/"src"; DATA=ROOT/"data"; RUN=ROOT/"runtime"; MODEL=ROOT/"model"; OUT=ROOT/"outputs"
stage=os.environ["E005_STAGE"]
stats=RUN/"token_length_stats.json"
def py(script,*args):
    cmd=[sys.executable,str(SRC/script),*map(str,args)]
    print("RUN",json.dumps(cmd),flush=True)
    proc=subprocess.Popen(cmd,cwd=SRC)
    while proc.poll() is None:
        print(f"HEARTBEAT stage={stage} pid={proc.pid}",flush=True);time.sleep(30)
    if proc.returncode: raise subprocess.CalledProcessError(proc.returncode,cmd)

if stage=="token_stats":
    py("token_stats_e005.py","--data-dir",DATA,"--output",stats)
elif stage=="benchmark":
    py("train_e005.py","--train-file",DATA/"train_pairs.parquet","--token-stats",stats,
       "--output-root",MODEL,"--benchmark-only","--benchmark-output",RUN/"benchmark.json",
       "--physical-batch-size",os.getenv("E005_BATCH","16"),"--inference-batch-size",os.getenv("E005_INFER_BATCH","64"))
elif stage=="train1":
    py("train_e005.py","--train-file",DATA/"train_pairs.parquet","--token-stats",stats,
       "--output-root",MODEL,"--stop-after-epoch","1","--physical-batch-size",os.getenv("E005_BATCH","16"),
       "--log-file",RUN/"training_log.csv")
elif stage=="infer_cal1":
    ml=str(json.loads(stats.read_text())["chosen_max_length"])
    py("infer_e005.py","--input",DATA/"calibration_pairs.parquet","--checkpoint",MODEL/"checkpoint-epoch1",
       "--output",OUT/"calibration_predictions_epoch1.parquet","--max-length",ml,"--batch-size",os.getenv("E005_INFER_BATCH","64"))
elif stage=="train2":
    py("train_e005.py","--train-file",DATA/"train_pairs.parquet","--token-stats",stats,
       "--output-root",MODEL,"--stop-after-epoch","2","--resume",MODEL/"checkpoint-epoch1",
       "--physical-batch-size",os.getenv("E005_BATCH","16"),"--log-file",RUN/"training_log.csv")
elif stage=="infer_cal2":
    ml=str(json.loads(stats.read_text())["chosen_max_length"])
    py("infer_e005.py","--input",DATA/"calibration_pairs.parquet","--checkpoint",MODEL/"checkpoint-epoch2",
       "--output",OUT/"calibration_predictions_epoch2.parquet","--max-length",ml,"--batch-size",os.getenv("E005_INFER_BATCH","64"))
elif stage=="infer_eval":
    epoch=os.environ["E005_EPOCH"];ml=str(json.loads(stats.read_text())["chosen_max_length"])
    py("infer_e005.py","--input",DATA/"evaluation_pairs.parquet","--checkpoint",MODEL/f"checkpoint-epoch{epoch}",
       "--output",OUT/"evaluation_predictions.parquet","--max-length",ml,"--batch-size",os.getenv("E005_INFER_BATCH","64"))
else: raise ValueError(stage)
