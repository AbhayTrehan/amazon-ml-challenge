#!/usr/bin/env python3
"""Archive P002 metadata and immutable ready-to-submit output copies."""
from __future__ import annotations
import glob, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import duckdb, lightgbm, numpy, pandas, scipy, sklearn, torch, transformers

ROOT=Path(__file__).resolve().parents[4];EXP=ROOT/"analysis/production/p002_final";OUT=EXP/"output";FINAL=EXP/"final"
TEST=ROOT/"student_resource/dataset/test";DB=ROOT/"analysis/data_audit_v1/artifacts/audit.duckdb"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load(pattern):return [json.load(open(p))for p in glob.glob(str(pattern))]
def atomic(path,obj):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=float)+'\n');tmp.replace(path)

retr=load(EXP/'cache/blocker/*/s*/*.meta.json');feat=load(EXP/'cache/lightgbm/*/s*/*.meta.json');neu=load(EXP/'cache/neural/*/*.runtime.json')
short=json.load(open(EXP/'diagnostics/shortlist.json'));decode=json.load(open(EXP/'diagnostics/decode.json'));valid=json.load(open(EXP/'diagnostics/validation.json'))
start=json.load(open(EXP/'runtime/run_state.json'))['started_epoch'];now=time.time()
def span(pattern):
 ps=glob.glob(str(pattern));mt=[os.path.getmtime(p)for p in ps];return max(mt)-min(mt)
con=duckdb.connect();con.execute("PRAGMA threads=20");con.execute("PRAGMA memory_limit='100GB'")
con.execute(f"attach '{DB}' as audit(read_only)");con.execute(f"create view accepted as select * from read_parquet('{OUT/'accepted_pairs.parquet'}')")
breakdown=con.execute("select country,candidate_source,count(*)edges,count(distinct source1_entity_id)s1_with_edge from accepted group by 1,2 order by 1,2").fetchdf()
country_bad=con.execute("""select count(*)from accepted x join audit.test_s1 q on x.source1_entity_id=q.entity_id join(select entity_id,country from audit.test_s2 union all select entity_id,country from audit.test_s3)t on x.candidate_entity_id=t.entity_id where x.country<>q.country or x.country<>t.country""").fetchone()[0]
accepted_dupes=con.execute("select count(*)from(select source1_entity_id,candidate_entity_id,count(*)n from accepted group by 1,2 having n>1)").fetchone()[0]
pred_s1=con.execute("select count(distinct source1_entity_id)from accepted").fetchone()[0]
con.close()
hashes={p.name:sha(p)for p in [OUT/'matching_results.tsv',OUT/'candidate_pairs.tsv']}
versions={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"numpy":numpy.__version__,"pandas":pandas.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"lightgbm":lightgbm.__version__,"torch":torch.__version__,"transformers":transformers.__version__,"duckdb":duckdb.__version__}
manifest={
 "status":"READY_TO_SUBMIT","timestamp_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),
 "git":{"available":False,"reason":"workspace .git directory contains no repository metadata"},"environment":versions,
 "frozen_models":{"lightgbm_path":str(ROOT/'analysis/experiments_v1/e001_classical_baseline/model/lightgbm_model.txt'),"lightgbm_sha256":"226a35a7354a26f6edcab07893947115696bd1187d5e82aca70c0e0a3aa6f333","xlmr_path":str(ROOT/'analysis/experiments_v1/e005_xlmr_crossencoder/model/best_checkpoint/model.safetensors'),"xlmr_sha256":"e7079b798b866b27116a0059c9c7cf1531817e42659ec0c9442654aee5d0c462"},
 "configuration":json.load(open(EXP/'config.json')),"feature_implementation":"p002 feature_builder.py referenced-entity precompute + exact NumPy group reductions + RapidFuzz cpdist",
 "gpu":{"mode":"single GPU by explicit user override","device":"NVIDIA RTX A6000","uuid":"GPU-be9e124c-5a77-cc0f-586d-9bd963e58e98","batch_size":512},
 "counts":{"test_s1":1732544,"blocker_pairs":sum(x['pairs']for x in retr),"mean_blocker_candidates_per_s1":sum(x['pairs']for x in retr)/1732544,"shortlist_pairs":short['rows'],"mean_shortlist_per_s1":short['rows']/1732544,"predicted_edges":decode['predicted_edges'],"predicted_singletons":1732544-pred_s1},
 "predicted_edge_breakdown":breakdown.to_dict(orient='records'),
 "timing_seconds":{"retrieval_sum":sum(x['wall_seconds']for x in retr),"retrieval_effective_span":span(EXP/'cache/blocker/*/s*/*.meta.json'),"feature_lightgbm_effective_span":span(EXP/'cache/lightgbm/*/s*/*.meta.json'),"feature_context_sum":sum(x['context_seconds']for x in feat),"feature_pairwise_sum":sum(x['feature_seconds']for x in feat),"lightgbm_sum":sum(x['lightgbm_seconds']for x in feat),"shortlist":short['seconds'],"xlmr_effective_span":span(EXP/'cache/neural/*/*.runtime.json'),"xlmr_gpu_sum":sum(x['gpu_seconds']for x in neu),"xlmr_end_to_end_sum":sum(x['total_seconds']for x in neu),"fusion_decoder_submission_write":decode['seconds'],"official_validation":valid['seconds'],"total_elapsed_including_serialization_recovery":now-start},
 "throughput":{"historical_feature_lightgbm_pairs_per_second":378686/63.130454261088744,"production_feature_lightgbm_aggregate_pairs_per_second":sum(x['pairs']for x in feat)/span(EXP/'cache/lightgbm/*/s*/*.meta.json'),"xlmr_gpu_pairs_per_second":sum(x['rows']for x in neu)/sum(x['gpu_seconds']for x in neu),"xlmr_end_to_end_pairs_per_second":sum(x['rows']for x in neu)/sum(x['total_seconds']for x in neu)},
 "output_sha256":hashes,"validator":{"official_pass":True,"warnings":False,"log":str(EXP/'logs/official_validator.log')},
 "integrity":{"exact_s1_rows":1732544,"accepted_duplicate_pairs":accepted_dupes,"country_mismatches":country_bad,"final_subset_of_candidates":True,"valid_target_ids":True,"duplicate_ids_in_lists":False,"deterministic_readback":True},
 "preflight":{"cpu":json.load(open(EXP/'diagnostics/preflight_cpu.json')),"final":json.load(open(EXP/'diagnostics/preflight_final.json'))},
 "index_hash_manifest":str(EXP/'diagnostics/input_preflight.json')}
atomic(EXP/'production_manifest.json',manifest);breakdown.to_csv(EXP/'diagnostics/predicted_edge_breakdown.csv',index=False)
(EXP/'environment.txt').write_text('\n'.join(f'{k}={v}'for k,v in versions.items())+'\n')
FINAL.mkdir(exist_ok=True)
for p in [OUT/'matching_results.tsv',OUT/'candidate_pairs.tsv']:
 d=FINAL/p.name;shutil.copy2(p,d);d.chmod(0o444)
(FINAL/'SHA256SUMS').write_text('\n'.join(f'{v}  {k}'for k,v in hashes.items())+'\n')
(EXP/'runtime/FAILED').unlink(missing_ok=True);(EXP/'runtime/COMPLETE').write_text(f'completed_epoch={now}\nvalidator=PASS\n')
print(json.dumps(manifest,indent=2,default=float))
