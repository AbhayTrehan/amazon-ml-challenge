#!/usr/bin/env python3
"""One-shot full-validation reverse-depth experiment.

The forward side is the already-complete R006B K20 pool.  This program only
retrieves reverse K5, then derives F20/R1,R2,R3,R5 from that single pool.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sparse_dot_topn import sp_matmul_topn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_b2_full as b2  # noqa: E402
from cache_utils import atomic_json, complete_manifest, content_key, sha256_file, validate_complete  # noqa: E402

REPORT = HERE / "reports/reverse_depth_full_validation"
OLD_RAW = HERE / "candidates/94d0af7bc51de525dc079a32ff26d0aa4fd59955723c6e72e4f8b37b9e2206f2"
OLD_B2 = HERE / "candidates/22dbcf8707c8739f3025264446e9734e2db0e3fc4024b023d8da27d4bf382129"
DEPTHS = (1, 2, 3, 5)
THREADS = 20
RETRIEVAL_VERSION = "full_owner_reverse_top5_v1"


def log(event: str, **values) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {event} " + json.dumps(values, sort_keys=True, default=float), flush=True)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def top5_identity(info: dict, forward_keys: dict, reverse_keys: dict) -> dict:
    aux = {}
    for country in b2.POP_COUNTRIES["clean_validation"]:
        for source in (2, 3):
            manifest = json.loads((b2.INDEXES / "b2_forward_refs/clean_validation" / country / f"s{source}.manifest.json").read_text())
            aux[f"{country}_s{source}"] = {
                "index_key": forward_keys[f"{country}_s{source}"],
                "aux_ids_sha256": manifest["identity"]["aux_ids_sha256"],
                "aux_file_sha256": manifest["identity"]["files"]["aux.parquet"],
            }
    return {
        "kind": "reverse_retrieval_pool",
        "version": RETRIEVAL_VERSION,
        "owner_population_key": info["manifest"]["content_key"],
        "owner_logical_ids_sha256": info["manifest"]["identity"]["logical_ids_sha256"],
        "owner_count": 431_284,
        "auxiliary_corpora": aux,
        "reverse_index_keys": reverse_keys,
        "configuration": {
            "country_partitioning": "literal string equality",
            "channels": ["s2_name", "s2_address", "s3_name", "s3_address"],
            "top_k": 5,
            "threshold": 1e-8,
            "sort": True,
            "implementation": "sparse_dot_topn.sp_matmul_topn",
        },
        "retrieval_code_version": RETRIEVAL_VERSION,
    }


def retrieve_top5(info: dict, forward_keys: dict, reverse_keys: dict) -> tuple[Path, dict, bool]:
    identity = top5_identity(info, forward_keys, reverse_keys)
    base = b2.CANDIDATES / content_key(identity)
    manifest_path = base / "manifest.json"
    cached = validate_complete(manifest_path)
    if cached:
        log("REVERSE_TOP5_CACHE_HIT", key=cached["content_key"], shards=cached["metadata"]["shards"])
        return base, cached, True
    started_all = time.perf_counter()
    runtime = []
    for country in b2.POP_COUNTRIES["clean_validation"]:
        for channel in ("name", "address"):
            ridx = b2.INDEXES / reverse_keys[f"{country}_{channel}"]
            corpus = pd.read_parquet(ridx / "corpus.parquet")
            s1rows = corpus.qrow.to_numpy(np.int32)
            matrix = sparse.load_npz(ridx / "matrix.npz").tocsr()
            idf = np.load(ridx / "idf.npy")
            for source in (2, 3):
                idx = b2.forward_index("clean_validation", country, source)
                aux = b2.normalized_aux("clean_validation", country, source, idx)
                apos = aux.apos.to_numpy(np.int32)
                for part, start in enumerate(range(0, len(aux), b2.AUX_SHARD)):
                    out = base / "shards" / country / f"s{source}_{channel}_{part:05d}.parquet"
                    meta = out.with_suffix(".meta.json")
                    if out.exists() and meta.exists():
                        runtime.append(json.loads(meta.read_text()))
                        continue
                    block = aux.iloc[start:start + b2.AUX_SHARD]
                    t0 = time.perf_counter()
                    xq = b2.qmatrix(block, channel, idf)
                    top = sp_matmul_topn(xq, matrix.T, top_n=5, threshold=1e-8, sort=True, n_threads=THREADS).tocsr()
                    local = np.repeat(np.arange(top.shape[0]), np.diff(top.indptr))
                    ranks = np.concatenate([np.arange(1, n + 1, dtype=np.int16) for n in np.diff(top.indptr)]) if top.nnz else np.empty(0, np.int16)
                    frame = pd.DataFrame({
                        "country_code": np.int8(b2.COUNTRY_CODE[country]),
                        "qrow": s1rows[top.indices],
                        "source": np.int8(source),
                        "apos": apos[start:start + len(block)][local],
                        "channel": np.int8(0 if channel == "name" else 1),
                        "rank": ranks,
                        "score": top.data.astype(np.float32),
                    })
                    atomic_parquet(frame, out)
                    row = {"country": country, "source": source, "channel": channel, "part": part,
                           "queries": len(block), "rows": len(frame), "wall_seconds": time.perf_counter() - t0}
                    atomic_json(meta, row)
                    runtime.append(row)
                    log("REVERSE_TOP5_SHARD_READY", **row)
                    del block, xq, top, local, ranks, frame
                    gc.collect()
            del matrix, corpus
            gc.collect()
    runtime_path = base / "runtime.json"
    atomic_json(runtime_path, {"stages": runtime, "wall_seconds": time.perf_counter() - started_all})
    artifacts = sorted((base / "shards").glob("*/*.parquet")) + sorted((base / "shards").glob("*/*.meta.json")) + [runtime_path]
    manifest = complete_manifest(manifest_path, identity, artifacts, {
        "query_count": int(sum(x["queries"] for x in runtime)),
        "row_count": int(sum(x["rows"] for x in runtime)),
        "shards": len([x for x in artifacts if x.suffix == ".parquet"]),
        "wall_seconds": time.perf_counter() - started_all,
        "schema": "integer-coded country/qrow/source/apos/channel/rank/score",
    })
    log("REVERSE_TOP5_COMPLETE", key=manifest["content_key"], **manifest["metadata"])
    return base, manifest, False


def score(tp, true):
    tp = np.asarray(tp, dtype=float)
    true = np.asarray(true, dtype=float)
    denominator = 1.25 * tp + .25 * (true - tp)
    result = np.divide(1.25 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    result[(true == 0) & (tp == 0)] = 1.0
    return result


def prepare_database(info: dict, pool: Path) -> tuple[duckdb.DuckDBPyConnection, Path]:
    REPORT.mkdir(parents=True, exist_ok=True)
    dbpath = REPORT / "evaluation.duckdb"
    if dbpath.exists():
        dbpath.unlink()
    con = duckdb.connect(str(dbpath))
    con.execute("pragma threads=20")
    con.execute("pragma memory_limit='100GB'")
    con.execute(f"attach '{b2.DB}' as audit(read_only)")
    forward_glob = str(OLD_RAW / "forward/*/*.parquet")
    reverse_glob = str(pool / "shards/*/*.parquet")
    old_b2_glob = str(OLD_B2 / "shards/*/*.parquet")
    con.execute(f"""create table forward_pairs as select distinct country_code,qrow,source,apos
        from read_parquet('{forward_glob}', union_by_name=true)""")
    con.execute(f"""create table old_reverse as select distinct country_code,qrow,source,apos,channel,score
        from read_parquet('{OLD_RAW}/reverse/*/*.parquet', union_by_name=true)""")
    con.execute(f"""create table pool_top1 as select country_code,qrow,source,apos,channel,score
        from read_parquet('{reverse_glob}', union_by_name=true) where rank=1""")
    missing_old, old_score_disagreements, changed_top1, top_score_disagreements = con.execute("""with old_presence as(
          select o.country_code,o.qrow,o.source,o.apos,o.channel,o.score old_score,p.rank old_pool_rank,p.score old_pool_score
          from old_reverse o left join read_parquet(?) p using(country_code,qrow,source,apos,channel)),
        top as(select * from pool_top1)
        select count(*)filter(where old_pool_rank is null),
          count(*)filter(where old_pool_score is not null and old_score!=old_pool_score),
          count(*)filter(where t.qrow is not null and t.qrow!=o.qrow),
          count(*)filter(where t.qrow is not null and t.qrow!=o.qrow and t.score!=o.old_score)
        from old_presence o left join top t on t.country_code=o.country_code and t.source=o.source
          and t.apos=o.apos and t.channel=o.channel""", [reverse_glob]).fetchone()
    if missing_old or old_score_disagreements or top_score_disagreements:
        raise RuntimeError(f"reverse-top1 tie normalization invalid: missing_old={missing_old}, old_score_disagreements={old_score_disagreements}, top_score_disagreements={top_score_disagreements}")
    tie_rows = con.execute("""select case when o.qrow=t.qrow then 'same_owner' else 'different_owner_equal_top_score' end selection,
          coalesce(cast(o.old_pool_rank as varchar),'absent') old_owner_rank_in_top5,count(*) channel_queries
        from (select o.*,p.rank old_pool_rank from old_reverse o join read_parquet(?) p
              using(country_code,qrow,source,apos,channel)) o
        join pool_top1 t on t.country_code=o.country_code and t.source=o.source and t.apos=o.apos and t.channel=o.channel
        where o.qrow=t.qrow or o.score=t.score
        group by all order by all""", [reverse_glob]).fetchdf()
    tie_rows.to_csv(REPORT / "r1_tie_parity.csv", index=False)
    atomic_json(REPORT / "r1_tie_parity.json", {"channel_queries": int(con.execute("select count(*) from old_reverse").fetchone()[0]),
        "different_top1_owner_equal_score_queries": int(changed_top1), "old_winner_missing_from_top5": int(missing_old),
        "old_winner_score_disagreements": int(old_score_disagreements), "top1_tie_score_disagreements": int(top_score_disagreements),
        "rank_distribution": tie_rows.to_dict(orient="records")})
    con.execute(f"""create table normalized_reverse as
        select country_code,qrow,source,apos,channel,1::smallint rank,score from old_reverse
        union all
        select p.country_code,p.qrow,p.source,p.apos,p.channel,
          (row_number() over(partition by p.country_code,p.source,p.apos,p.channel order by p.rank)+1)::smallint rank,p.score
        from read_parquet('{reverse_glob}', union_by_name=true) p
        anti join old_reverse o using(country_code,qrow,source,apos,channel)""")
    con.execute(f"""create table reverse_pairs as select country_code,qrow,source,apos,
        min(rank) filter(channel=0)::smallint name_rank,
        min(rank) filter(channel=1)::smallint address_rank
        from normalized_reverse group by all""")
    con.execute(f"""create table old_r1_pairs as select distinct country_code,qrow,source,apos
        from read_parquet('{old_b2_glob}')""")
    con.execute("""create table pair_keys as
        select country_code,qrow,source,apos from forward_pairs
        union select country_code,qrow,source,apos from old_r1_pairs
        union select country_code,qrow,source,apos from reverse_pairs""")
    con.execute("""create table all_pairs as select k.country_code,k.qrow,k.source,k.apos,
        f.qrow is not null is_forward,o.qrow is not null is_old_r1,r.name_rank,r.address_rank,
        least(coalesce(r.name_rank,32767),coalesce(r.address_rank,32767))::smallint reverse_rank
        from pair_keys k left join forward_pairs f using(country_code,qrow,source,apos)
        left join old_r1_pairs o using(country_code,qrow,source,apos)
        left join reverse_pairs r using(country_code,qrow,source,apos)""")
    # Exact key parity is mandatory before any scoring.
    parity = con.execute("""with r1 as(select country_code,qrow,source,apos from all_pairs where is_forward or reverse_rank<=1),
        old as(select country_code,qrow,source,apos from old_r1_pairs)
        select (select count(*) from r1),(select count(*) from old),
          (select count(*) from(select * from r1 except select * from old)),
          (select count(*) from(select * from old except select * from r1))""").fetchone()
    atomic_json(REPORT / "r1_parity.json", {"derived_rows": parity[0], "existing_rows": parity[1], "derived_only": parity[2], "existing_only": parity[3]})
    if parity[2] or parity[3] or parity[0] != parity[1]:
        raise RuntimeError(f"STOP: R1 parity failure {parity}")
    log("R1_PARITY_EXACT", rows=parity[0])

    pop_path = info["path"]
    con.execute(f"""create table owners(source1_entity_id varchar,country varchar,true_matches integer,s2_matches integer,s3_matches integer);
        insert into owners select source1_entity_id,country,true_matches,s2_matches,s3_matches from read_parquet('{pop_path}')""")
    owner_rows = con.execute("select count(*),count(distinct source1_entity_id) from owners").fetchone()
    if owner_rows != (431284, 431284):
        raise RuntimeError(f"owner identity failed: {owner_rows}")
    pieces = []
    for country in b2.POP_COUNTRIES["clean_validation"]:
        qpath = b2.query_map("clean_validation", country, info)
        for source in (2, 3):
            # `apos` is intentionally reconstructed by the established loader;
            # it is not a physical column in all historical aux parquets.
            aux_frame = b2.normalized_aux("clean_validation", country, source,
                                          b2.forward_index("clean_validation", country, source))[["apos", "entity_id", "business_address"]]
            aux_name = f"auxmap_{country.lower()}_{source}"
            con.register(aux_name, aux_frame)
            pieces.append(f"""select {b2.COUNTRY_CODE[country]}::tinyint country_code,q.qrow::integer qrow,{source}::tinyint as \"source\",
                a.apos::integer apos,q.entity_id s1_id,a.entity_id candidate_id,
                (coalesce(a.business_address,'')='') address_missing,o.true_matches
                from read_parquet('{qpath}') q join owners o on o.source1_entity_id=q.entity_id
                join audit.edges e on e.s1=q.entity_id and e.target_source={source}
                join {aux_name} a on a.entity_id=e.target_id""")
    con.execute("create table truth as " + " union all ".join(pieces))
    expected_truth = con.execute("select sum(true_matches) from owners").fetchone()[0]
    actual_truth = con.execute("select count(*) from truth").fetchone()[0]
    if expected_truth != actual_truth:
        raise RuntimeError(f"truth/aux contract mismatch: expected {expected_truth}, got {actual_truth}")
    return con, dbpath


def evaluate(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configurations = []
    slices = []
    heavy = []
    per_depth = {}
    base_recall = base_oracle = base_mean = None
    base_hits = None
    for depth in DEPTHS:
        total_pairs, distinct_pairs = con.execute(f"""select count(*),count(distinct(country_code,qrow,source,apos))
            from all_pairs where is_old_r1 or reverse_rank<={depth}""").fetchone()
        duplicate_count = total_pairs - distinct_pairs
        if duplicate_count:
            raise RuntimeError(f"duplicate pairs at R{depth}: {duplicate_count}")
        con.execute(f"""create or replace temp table counts as select country_code,qrow,source,count(*)::integer candidates
            from all_pairs where is_old_r1 or reverse_rank<={depth} group by all""")
        con.execute(f"""create or replace temp table hits as select t.country_code,t.qrow,t.source,count(*)::integer tp
            from truth t join all_pairs p using(country_code,qrow,source,apos)
            where p.is_old_r1 or p.reverse_rank<={depth} group by all""")
        parts = []
        for country in b2.POP_COUNTRIES["clean_validation"]:
            qpath = b2.query_map("clean_validation", country, {"frame": None})
            parts.append(f"select {b2.COUNTRY_CODE[country]}::tinyint country_code,qrow,entity_id from read_parquet('{qpath}')")
        owner_map_sql = " union all ".join(parts)
        wide = con.execute(f"""with q as({owner_map_sql}),
            cw as(select country_code,qrow,max(candidates)filter(source=2)candidates_s2,max(candidates)filter(source=3)candidates_s3 from counts group by all),
            hw as(select country_code,qrow,max(tp)filter(source=2)tp_s2,max(tp)filter(source=3)tp_s3 from hits group by all),
            z as(select q.country_code,q.qrow,q.entity_id,coalesce(cw.candidates_s2,0)::integer candidates_s2,
              coalesce(cw.candidates_s3,0)::integer candidates_s3,coalesce(hw.tp_s2,0)::integer tp_s2,
              coalesce(hw.tp_s3,0)::integer tp_s3 from q left join cw using(country_code,qrow) left join hw using(country_code,qrow))
            select z.*,o.country,o.true_matches,o.s2_matches,o.s3_matches from z join owners o on o.source1_entity_id=z.entity_id""").fetchdf()
        if len(wide) != 431284:
            raise RuntimeError(f"R{depth}: owner count {len(wide)}")
        wide["candidates"] = wide.candidates_s2 + wide.candidates_s3
        wide["tp"] = wide.tp_s2 + wide.tp_s3
        wide["oracle"] = score(wide.tp, wide.true_matches)
        missing_by_s1 = con.execute("select s1_id,bool_or(address_missing) missing from truth group by 1").fetchdf().set_index("s1_id").missing
        wide["missing_address"] = wide.entity_id.map(missing_by_s1).fillna(False)
        edge_recall = float(wide.tp.sum() / wide.true_matches.sum())
        oracle = float(wide.oracle.mean())
        mean = float(wide.candidates.mean())
        if depth == 1:
            base_recall, base_oracle, base_mean = edge_recall, oracle, mean
        unique_tp = con.execute(f"""select count(*) from truth t join all_pairs p using(country_code,qrow,source,apos)
            where not p.is_old_r1 and p.reverse_rank<={depth}""").fetchone()[0]
        configurations.append({
            "reverse_k": depth, "config": f"F20_R{depth}", "s1": len(wide), "total_candidate_pairs": int(wide.candidates.sum()),
            "mean_candidates": mean, "median_candidates": float(wide.candidates.median()),
            "p95_candidates": float(wide.candidates.quantile(.95)), "p99_candidates": float(wide.candidates.quantile(.99)),
            "p999_candidates": float(wide.candidates.quantile(.999)), "max_candidates": int(wide.candidates.max()),
            "edge_recall": edge_recall, "oracle_macro_f05": oracle,
            "delta_edge_recall_vs_r1": edge_recall - base_recall, "delta_oracle_vs_r1": oracle - base_oracle,
            "extra_candidates_per_s1_vs_r1": mean - base_mean, "unique_true_edges_recovered_vs_r1": int(unique_tp),
        })

        def add_slice(name, frame, tp="tp", true="true_matches"):
            truth_total = frame[true].sum()
            slices.append({"reverse_k": depth, "slice": name, "n_s1": len(frame),
                           "candidate_pairs": int(frame.candidates.sum()), "mean_candidates": float(frame.candidates.mean()) if len(frame) else 0.0,
                           "edge_recall": float(frame[tp].sum() / truth_total) if truth_total else 1.0,
                           "oracle_macro_f05": float(score(frame[tp], frame[true]).mean()) if len(frame) else float("nan")})
        add_slice("overall", wide)
        for country, z in wide.groupby("country", sort=True):
            add_slice(country, z)
        add_slice("S2", wide, "tp_s2", "s2_matches")
        add_slice("S3", wide, "tp_s3", "s3_matches")
        add_slice("true_candidate_address_missing", wide[wide.missing_address])
        add_slice("true_candidate_address_present", wide[~wide.missing_address])
        for cardinality in range(5):
            add_slice(f"cardinality_{cardinality}", wide[wide.true_matches.eq(cardinality)])
        add_slice("cardinality_5+", wide[wide.true_matches.ge(5)])
        counts = wide.candidates.sort_values(ascending=False).to_numpy()
        total = counts.sum()
        row = {"reverse_k": depth}
        for threshold in (500, 1000, 5000, 10000, 50000):
            row[f"s1_gt_{threshold}"] = int((wide.candidates > threshold).sum())
        for label, fraction in (("top_1pct_pair_share", .01), ("top_0_1pct_pair_share", .001), ("top_0_01pct_pair_share", .0001)):
            row[label] = float(counts[:math.ceil(len(counts) * fraction)].sum() / total)
        heavy.append(row)
        per_depth[depth] = wide
        log("DEPTH_EVALUATED", **configurations[-1])
    return pd.DataFrame(configurations), pd.DataFrame(slices), pd.DataFrame(heavy), per_depth[1]


def missed_diagnostics(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    misses = con.execute("""select t.*,
        p.name_rank,p.address_rank,
        case when p.name_rank<=5 and p.address_rank<=5 then 'both'
             when p.name_rank<=5 then 'name_only' when p.address_rank<=5 then 'address_only' else 'not_recovered' end recoverable_by,
        case when least(coalesce(p.name_rank,32767),coalesce(p.address_rank,32767))<=2 then 'K2'
             when least(coalesce(p.name_rank,32767),coalesce(p.address_rank,32767))=3 then 'K3'
             when least(coalesce(p.name_rank,32767),coalesce(p.address_rank,32767))<=5 then 'K5'
             else '>5_or_not_recovered' end recovery_bucket
        from truth t left join all_pairs p using(country_code,qrow,source,apos)
        where not coalesce(p.is_old_r1,false)
        order by country_code,qrow,source,apos""").fetchdf()
    country_names = {v: k for k, v in b2.COUNTRY_CODE.items()}
    misses["country"] = misses.country_code.map(country_names)
    aggregate = []
    dimensions = {
        "overall": pd.Series(True, index=misses.index),
        "US": misses.country.eq("US"), "India": misses.country.eq("India"),
        "S2": misses.source.eq(2), "S3": misses.source.eq(3),
        "address_missing": misses.address_missing, "address_present": ~misses.address_missing,
    }
    for label, mask in dimensions.items():
        z = misses[mask]
        for channel, rank_col in (("either", None), ("name", "name_rank"), ("address", "address_rank")):
            if channel == "either":
                values = z.recovery_bucket
            else:
                ranks = z[rank_col].fillna(32767)
                values = pd.Series(np.select([ranks.le(2), ranks.eq(3), ranks.le(5)], ["K2", "K3", "K5"], default=">5_or_not_recovered"), index=z.index)
            counts = values.value_counts()
            aggregate.append({"slice": label, "channel": channel, "missed_r1_true_edges": len(z),
                              "recovered_k2": int(counts.get("K2", 0)), "recovered_k3_only": int(counts.get("K3", 0)),
                              "recovered_k5_only": int(counts.get("K5", 0)), ">5_or_not_recovered": int(counts.get(">5_or_not_recovered", 0))})
    contributions = []
    contribution_slices = {
        "overall": pd.Series(True, index=misses.index), "India": misses.country.eq("India"),
        "S3": misses.source.eq(3), "address_missing": misses.address_missing,
        "cardinality_1": misses.true_matches.eq(1),
        "India_S3_missing_cardinality1": misses.country.eq("India") & misses.source.eq(3) & misses.address_missing & misses.true_matches.eq(1),
    }
    for depth in (2, 3, 5):
        for slice_name, mask in contribution_slices.items():
            z = misses[mask]
            for channel, rank_col in (("S2_name", "name_rank"), ("S2_address", "address_rank"),
                                      ("S3_name", "name_rank"), ("S3_address", "address_rank")):
                source = int(channel[1])
                zz = z[z.source.eq(source)]
                contributions.append({"reverse_k": depth, "slice": slice_name, "channel": channel,
                                      "unique_true_edges_recovered_over_r1": int(zz[rank_col].fillna(32767).le(depth).sum())})
            contributions.append({"reverse_k": depth, "slice": slice_name, "channel": "union_all_reverse_channels",
                                  "unique_true_edges_recovered_over_r1": int(z[["name_rank", "address_rank"]].fillna(32767).min(axis=1).le(depth).sum())})
    return misses, pd.DataFrame(aggregate), pd.DataFrame(contributions)


def report_results(metrics: pd.DataFrame, slices: pd.DataFrame, heavy: pd.DataFrame,
                   miss_summary: pd.DataFrame, contributions: pd.DataFrame, pool_manifest: dict,
                   cache_hit: bool, retrieval_seconds: float, eval_seconds: float) -> str | None:
    qualifying = metrics[metrics.oracle_macro_f05.ge(.995)].sort_values("reverse_k")
    freeze = None if qualifying.empty else f"F20_R{int(qualifying.iloc[0].reverse_k)}"
    old_owners, full_owners = 50_000, 431_284
    old_cache_manifest = json.loads((HERE / "candidates/0ac9ca0cfd02622416c823b1f4d5121f94872b4e7c56c8aacd099159a5b5b995/manifest.json").read_text())
    old_raw_queries = int(old_cache_manifest["metadata"]["query_count"] * 2)
    full_raw_queries = int(pool_manifest["metadata"]["query_count"])
    old_ratio, full_ratio = old_raw_queries / old_owners, full_raw_queries / full_owners
    tie_info = json.loads((REPORT / "r1_tie_parity.json").read_text())
    lines = ["# EXECUTIVE_RESULT", ""]
    for row in metrics.itertuples():
        lines.append(f"R{row.reverse_k}_ORACLE = {row.oracle_macro_f05:.10f}")
    lines.append("")
    for row in metrics.itertuples():
        lines.append(f"R{row.reverse_k}_MEAN_CANDIDATES = {row.mean_candidates:.6f}")
    lines.extend(["", f"BLOCKER_CANDIDATE_TO_FREEZE = {freeze}" if freeze else "NO_REVERSE_K_LE5_REACHED_GATE", "",
                  "# WHY_50K_WAS_OPTIMISTIC", "",
                  f"The old reverse owner corpus contained {old_owners:,} S1; the authoritative full corpus contains {full_owners:,}. "
                  f"The old run made {old_raw_queries:,} raw source/channel auxiliary queries ({old_ratio:.3f} per owner); the full run made {full_raw_queries:,} ({full_ratio:.3f} per owner). "
                  "Restricting the owners both concentrated reverse assignments onto fewer benchmark S1 (raising candidates/S1) and removed 381,284 competing owners, making true-owner top ranks easier. Forward K20 was unaffected.", "",
                  "# REVERSE_DEPTH_CURVE", "", "| Reverse K | Mean candidates/S1 | Edge recall | Oracle F0.5 | Delta oracle vs R1 | Unique TP recovered |", "|---:|---:|---:|---:|---:|---:|"])
    for row in metrics.itertuples():
        lines.append(f"| {row.reverse_k} | {row.mean_candidates:.6f} | {row.edge_recall:.9f} | {row.oracle_macro_f05:.9f} | {row.delta_oracle_vs_r1:+.9f} | {row.unique_true_edges_recovered_vs_r1:,} |")
    lines.extend(["", "# MISSED_TRUE_EDGE_RANKS", "", "The detailed edge-level artifact records independent name/address ranks. Aggregate either-channel recovery:", "",
                  "| Slice | R1 missed edges | K2 | K3 only | K5 only | >5/not recovered |", "|---|---:|---:|---:|---:|---:|"])
    for _, row in miss_summary[miss_summary.channel.eq("either")].iterrows():
        lines.append(f"| {row['slice']} | {int(row['missed_r1_true_edges']):,} | {int(row['recovered_k2']):,} | {int(row['recovered_k3_only']):,} | {int(row['recovered_k5_only']):,} | {int(row['>5_or_not_recovered']):,} |")
    lines.extend(["", "# CHANNEL_CONTRIBUTIONS", "", "See `reverse_channel_contributions.csv`; counts are diagnostic and channel rows can overlap when both name and address recover an edge.", "",
                  "# HEAVY_TAIL", "", "| K | >500 | >1k | >5k | >10k | >50k | Top 1% share | Top 0.1% | Top 0.01% |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in heavy.itertuples():
        lines.append(f"| {row.reverse_k} | {row.s1_gt_500} | {row.s1_gt_1000} | {row.s1_gt_5000} | {row.s1_gt_10000} | {row.s1_gt_50000} | {row.top_1pct_pair_share:.6f} | {row.top_0_1pct_pair_share:.6f} | {row.top_0_01pct_pair_share:.6f} |")
    lines.extend(["", "# CACHE_INTEGRITY", "",
                  f"- Reverse top5 content key: `{pool_manifest['content_key']}`.",
                  f"- Cache hit on this execution: `{str(cache_hit).lower()}`; retrieval wall time recorded by this invocation: {retrieval_seconds:.2f}s.",
                  "- Lower depths derive from the single top5 pool. Per channel, the existing top1 owner is placed first, followed by the other cached top5 hits in score order.",
                  f"- Across {tie_info['channel_queries']:,} channel queries, {tie_info['different_top1_owner_equal_score_queries']:,} top1 choices differed; original winners were present in cached top5 at matching scores. R1 pair parity is exact.",
                  "- Canonical owner count is 431,284; the owner population content key is `401278452d9987efdf8d5cdeb52c08211b291bb19ad435362154f09e15177352`.",
                  "- `.git` is empty, so no Git HEAD is available.", "",
                  "# NEXT_DECISION", ""])
    if freeze:
        row = qualifying.iloc[0]
        lines.append(f"Freeze `{freeze}`: it is the lowest tested reverse K meeting oracle >= 0.9950 (mean {row.mean_candidates:.3f} candidates/S1). Do not launch train/test retrieval in this experiment.")
    else:
        lines.append("No tested K <= 5 met 0.9950. Stop here; do not test additional retrieval variants automatically.")
    lines.extend(["", f"Evaluation/reporting wall time: {eval_seconds:.2f}s.", ""])
    (REPORT / "REPORT.md").write_text("\n".join(lines))
    return freeze


def main() -> None:
    overall_start = time.perf_counter()
    pops = b2.populations()
    info = pops["clean_validation"]
    if len(info["frame"]) != 431284 or sha256_file(b2.VAL_IDS) != "5084362ccd992845ddf65508293552b2919b8a4f1c13190e82e7bf026ee5e74f":
        raise RuntimeError("canonical validation population mismatch")
    forward_keys, reverse_keys = b2.register_indexes("clean_validation", info)
    retrieval_start = time.perf_counter()
    pool, pool_manifest, cache_hit = retrieve_top5(info, forward_keys, reverse_keys)
    retrieval_seconds = time.perf_counter() - retrieval_start
    eval_start = time.perf_counter()
    con, dbpath = prepare_database(info, pool)
    metrics, slices, heavy, _ = evaluate(con)
    misses, miss_summary, contributions = missed_diagnostics(con)
    con.close()
    atomic_parquet(misses, REPORT / "reverse_rank_misses.parquet")
    miss_summary.to_csv(REPORT / "reverse_rank_misses.csv", index=False)
    metrics.to_csv(REPORT / "config_metrics.csv", index=False)
    slices.to_csv(REPORT / "slice_metrics.csv", index=False)
    contributions.to_csv(REPORT / "reverse_channel_contributions.csv", index=False)
    heavy.to_csv(REPORT / "heavy_tail_metrics.csv", index=False)
    eval_seconds = time.perf_counter() - eval_start
    freeze = report_results(metrics, slices, heavy, miss_summary, contributions, pool_manifest, cache_hit, retrieval_seconds, eval_seconds)
    if freeze:
        frozen_identity = {"kind": "frozen_blocker_config", "definition": freeze, "forward_k": 20,
                           "reverse_k": int(freeze.rsplit("R", 1)[1]), "reverse_pool_key": pool_manifest["content_key"],
                           "population_key": info["manifest"]["content_key"], "decision_gate": "lowest K in [1,2,3,5] with full oracle >= 0.9950"}
        frozen_path = HERE / "manifests" / f"{freeze.lower()}_blocker_config.json"
        atomic_json(frozen_path, {"state": "COMPLETE", "content_key": content_key(frozen_identity), "identity": frozen_identity})
    artifacts = [REPORT / name for name in ("REPORT.md", "config_metrics.csv", "slice_metrics.csv", "reverse_rank_misses.csv",
                 "reverse_rank_misses.parquet", "reverse_channel_contributions.csv", "heavy_tail_metrics.csv", "r1_parity.json")]
    artifacts.extend([REPORT / "r1_tie_parity.csv", REPORT / "r1_tie_parity.json"])
    identity = {"kind": "reverse_depth_full_validation_report", "reverse_pool_key": pool_manifest["content_key"],
                "existing_b2_key": OLD_B2.name, "depths": list(DEPTHS), "metric": "official grouped macro F0.5 oracle", "code_sha256": sha256_file(Path(__file__))}
    complete_manifest(REPORT / "manifest.json", identity, artifacts, {"freeze": freeze, "wall_seconds": time.perf_counter() - overall_start})
    # Cache replay check without re-running retrieval: validates all checksums and identity.
    replay = validate_complete(pool / "manifest.json")
    if not replay or replay["content_key"] != pool_manifest["content_key"]:
        raise RuntimeError("reverse pool cache replay validation failed")
    (REPORT / "COMPLETE").write_text(pool_manifest["content_key"] + "\n")
    log("EXPERIMENT_COMPLETE", freeze=freeze, report=str(REPORT / "REPORT.md"), total_seconds=time.perf_counter() - overall_start)


if __name__ == "__main__":
    main()
