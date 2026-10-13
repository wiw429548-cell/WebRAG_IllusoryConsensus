"""RQ1 / RQ2: run every RAG method with one reader LLM on EchoBench (all grid conditions) or RAMDocs.

python -m indierag.run_rag --reader Qwen/Qwen2.5-7B-Instruct --dataset echobench
-> results/rag/<reader>/<dataset>/{metrics.json, preds/<method>.jsonl}
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict

import numpy as np

from .claims import extract_claims
from .conditions import doc_index, make_instances
from .dependence import DependenceModel
from .llm import LLM
from .methods import Ctx, run_method
from .signals import DocStore
from .utils import (ROOT, DATA, contains, em, f1, get_logger, load_config, load_json, normalize_answer, read_jsonl,
                    save_json, slug, write_jsonl)

ALL_METHODS = ["vanilla", "dedup", "semdedup", "majority", "indievote:lex", "indievote:cos", "astute", "instructrag",
               "trustrag", "madam", "cag", "indiebayes:none",
               "indievote:H", "indievote:L", "indievote:LLM", "indievote:oracle", "indiebayes:L",
               "indierag:H", "indierag:L", "indierag:L+meta", "indierag:LLM", "indierag:oracle",
               "indierag_table:L", "indierag_compress:L"]
GROUPS = "main,types,indep,ranked,noisy,style,symC"


def load_instances(cfg, dataset, groups, max_questions=None):
    if dataset == "echobench":
        idx = doc_index()
        insts = make_instances(cfg, "test", groups, max_questions=max_questions)
        for it in insts:
            it["docs"] = [idx[d] for d in it["doc_ids"]]
            it["golds"] = it["aliases"]
            it["wrongs"] = [it["wrong"]]
        return insts
    rows = read_jsonl(DATA / "raw" / "ramdocs" / "RAMDocs_test.jsonl")[:cfg.get("ramdocs_max") or None]
    insts = []
    for i, r in enumerate(rows):
        docs = [{"text": d["text"], "origin": f"r{i}:{j}", "type": d.get("type")} for j, d in enumerate(r["documents"])]
        insts.append({"iid": f"ramdocs{i}", "question": r["question"], "docs": docs, "cond": "ramdocs", "setting": "R",
                      "golds": r.get("gold_answers", []), "wrongs": r.get("wrong_answers", []), "m": 0, "k": 0})
    return insts


def score(it, pred):
    s = {"acc": contains(pred, it["golds"]), "wrong": contains(pred, it["wrongs"]) if it["wrongs"] else 0.0,
         "em": em(pred, it["golds"]), "f1": f1(pred, it["golds"])}
    s["acc_strict"] = float(s["acc"] and not s["wrong"])  # hedging with both answers does not count as correct
    if it["setting"] == "R":  # RAMDocs: all valid answers and no misinformation answer
        # RAMDocs answers often carry units ("3,559 people"): a predicted part matches a gold answer if one contains
        # the other or their token-F1 >= 0.5
        parts = [x for x in pred.replace(" and ", ";").split(";") if normalize_answer(x)]

        def soft(g):
            return contains(pred, [g]) or any(contains(g, [x]) or f1(x, [g]) >= 0.5 for x in parts)
        golds = it["golds"]
        s["acc"] = float(any(soft(g) for g in golds))
        s["wrong"] = float(any(soft(w) for w in it["wrongs"])) if it["wrongs"] else 0.0
        s["acc_strict"] = float(s["acc"] and not s["wrong"])
        s["acc_all"] = float(all(soft(g) for g in golds) and not s["wrong"]) if golds else 0.0
    ic, iw = it.get("n_indep_correct"), it.get("n_indep_wrong")
    if ic is not None and ic != iw:  # agreement with the answer backed by MORE INDEPENDENT origins
        s["indep_majority_agree"] = s["wrong"] if iw > ic else s["acc"]
    return s


def method_scope(cfg, m, dataset):
    """condition groups / names a method is run on (None = every instance); see `rag.scope` in the config"""
    if dataset != "echobench":
        return None
    sc = (cfg.get("rag") or {}).get("scope") or {}
    key = m if m in sc else (m.split("#")[0] + "#" if "#" in m else None)
    return set(sc[key]) if key in sc else None


def method_var(m):
    return m.split(":")[1].split("#")[0] if ":" in m else None


def resolve_tp(arg, model):
    """--tp all = every visible GPU (the scheduler gives a 2-GPU job both GPUs); auto = all for >=30B models"""
    import torch
    n = max(1, torch.cuda.device_count())  # NVML count; unlike is_available() it does not initialise CUDA before vLLM
    if arg == "all" or (arg == "auto" and any(x in model for x in ("32B", "70B", "72B"))):
        return n
    return 1 if arg == "auto" else max(1, min(int(arg), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--dataset", default="echobench", choices=["echobench", "ramdocs"])
    ap.add_argument("--methods", default=None, help="comma list (default: all methods + ablations + sensitivity)")
    ap.add_argument("--groups", default=GROUPS)
    ap.add_argument("--max_questions", type=int, default=None, help="default: rag.n_test_questions of the config")
    ap.add_argument("--tp", default="auto", help="tensor parallel size: auto | all | <int>")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = ROOT / "results" / "rag" / (slug(a.reader) + a.tag) / a.dataset
    log = get_logger(f"rag/{slug(a.reader)}/{a.dataset}", out / "run.log")
    nq = a.max_questions if a.max_questions is not None else (cfg.get("rag") or {}).get("n_test_questions")
    insts = load_instances(cfg, a.dataset, a.groups.split(","), max_questions=nq)
    log.info(f"{len(insts)} instances ({len({it.get('qid', it['iid']) for it in insts})} questions)")
    mc = cfg["models"]
    tp = resolve_tp(a.tp, a.reader)
    llm = LLM(a.reader, max_model_len=mc["max_model_len"], gpu_mem=mc["gpu_mem"], tp=tp)
    log.info(f"engine: {llm.engine_info}")
    store = DocStore()
    for it in insts:  # retriever order for the 'ranked' group
        if it.get("order") == "ranked":
            qv = store.query_emb(it["question"])
            if qv is not None:
                it["docs"] = sorted(it["docs"], key=lambda d: -float(store.get(d["text"])[0] @ qv))
    # ---- per-document claims (cached in the LLM cache)
    u0 = llm.usage_snapshot()
    pairs = sorted({(it["question"], d["text"]) for it in insts for d in it["docs"]})
    t0 = time.time()
    claims = dict(zip(pairs, extract_claims(llm, pairs)))
    log.info(f"claims for {len(pairs)} (question, doc) pairs in {time.time() - t0:.0f}s")
    u1 = llm.usage_snapshot()
    claim_cost = {k: u1[k] - u0[k] for k in u1}
    claim_acc = defaultdict(list)  # error analysis of the per-document claim extraction (EchoBench roles are known)
    for it in insts:
        for d in it["docs"]:
            role = d.get("role")
            c = claims.get((it["question"], d["text"]), "")
            if role in ("correct", "correct_para", "correct_copy"):
                claim_acc[role].append(contains(c, it["golds"]))
            elif role in ("origin", "copy", "indep_wrong"):
                claim_acc[role].append(contains(c, it["wrongs"]))
            elif role == "distractor":
                claim_acc[role].append(float(c == ""))
    if a.methods:
        methods = a.methods.split(",")
    else:
        methods = list(ALL_METHODS)
        if a.dataset == "echobench":  # clustering-threshold sensitivity + end-to-end feature ablations
            methods += [f"indierag:L#{t}" for t in cfg["dependence"].get("threshold_sensitivity", [])]
            methods += list((cfg.get("rag") or {}).get("ablation_votes") or [])
    if a.dataset == "ramdocs":  # no origin labels on RAMDocs; thresholds / ablations only on EchoBench
        methods = [m for m in methods if not m.endswith("oracle") and "#" not in m and "[" not in m]
    variants = {method_var(m) for m in methods if ":" in m}
    dep = {v: DependenceModel(store, v, llm=llm if v == "LLM" else None, cfg=cfg) for v in variants | {"none"}}
    ctx = Ctx(llm, store, dep, claims, cfg, multi=(a.dataset == "ramdocs"))
    dep_cost = defaultdict(lambda: {"sec": 0.0, "n": 0, "calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    cache, metrics = {}, {}
    (out / "method_metrics").mkdir(parents=True, exist_ok=True)
    for m in ["vanilla"] + [x for x in methods if x != "vanilla"]:
        scope = method_scope(cfg, m, a.dataset)
        sub = [it for it in insts if scope is None or it.get("group") in scope or it["cond"] in scope]
        if not sub:
            log.info(f"{m}: no instance in scope {scope} -> skipped")
            continue
        sig = hashlib.sha1(json.dumps([it["iid"] for it in sub]).encode()).hexdigest()
        mfile = out / "method_metrics" / f"{m.replace(':', '_')}.json"
        pfile = out / "preds" / f"{m.replace(':', '_')}.jsonl"
        if mfile.exists() and pfile.exists():  # method-level resume (the LLM cache makes re-runs cheap anyway)
            saved = load_json(mfile)
            if saved.get("_sig") == sig:
                metrics[m] = saved["metrics"]
                if m == "vanilla":
                    cache["vanilla_by_iid"] = {r["iid"]: (r["raw"], r["pred"]) for r in read_jsonl(pfile)}
                log.info(f"{m:22s} resumed from {mfile.name}")
                continue
        v = method_var(m)
        if v and v != "none":  # dependence estimation, timed separately (LLM judge tokens are attributed here)
            todo = [it for it in sub if (it["iid"], v) not in ctx._P]
            us, t1 = llm.usage_snapshot(), time.time()
            for it in todo:
                ctx.P(it, v)
            ue = llm.usage_snapshot()
            dc = dep_cost[v]
            dc["sec"] += time.time() - t1
            dc["n"] += len(todo)
            for k in ("calls", "prompt_tokens", "completion_tokens"):
                dc[k] += ue[k] - us[k]
        t0 = time.time()
        us = llm.usage_snapshot()
        res = run_method(m, sub, ctx, cache)
        ue = llm.usage_snapshot()
        if m == "vanilla":
            cache["vanilla_by_iid"] = {it["iid"]: r for it, r in zip(sub, res)}
        rows, agg = [], defaultdict(lambda: defaultdict(list))
        for it, (raw, pred) in zip(sub, res):
            s = score(it, pred)
            rows.append({"iid": it["iid"], "qid": it.get("qid", it["iid"]), "cond": it["cond"], "pred": pred,
                         "raw": raw[:2000], "roles": [d.get("role") or d.get("type") for d in it["docs"]], **s})
            for k, val in s.items():
                agg[it["cond"]][k].append(val)
                agg["_all"][k].append(val)
        metrics[m] = {c: {k: float(np.mean(x)) for k, x in d.items()} | {"n": len(d["acc"])} for c, d in agg.items()}
        metrics[m]["_time_sec"] = time.time() - t0
        uses_claims = m in ("majority",) or m.split(":")[0] in ("indievote", "indiebayes", "indierag", "indierag_table",
                                                                "indierag_compress")
        cost = {k: ue[k] - us[k] for k in ue}
        if uses_claims:  # per-document claim extraction is part of these methods' cost: counted per instance as if
            # stand-alone (#docs of the instance x mean cost of one claim call), although it is shared across conditions
            n_doc = sum(len(it["docs"]) for it in sub)
            cost = {k: cost[k] + n_doc * claim_cost[k] / max(1, len(pairs)) for k in cost}
        if v and v in dep_cost and dep_cost[v]["n"]:  # amortised dependence-estimation cost (e.g. LLM pair judge)
            dc = dep_cost[v]
            cost = {k: cost[k] + len(sub) * dc.get(k, 0) / dc["n"] for k in cost}
        metrics[m]["_cost_per_instance"] = {k: x / max(1, len(sub)) for k, x in cost.items()}
        if v and v in dep_cost and dep_cost[v]["n"]:
            metrics[m]["_dep_sec_per_instance"] = dep_cost[v]["sec"] / dep_cost[v]["n"]
        write_jsonl(rows, pfile)
        save_json({"_sig": sig, "metrics": metrics[m]}, mfile)
        log.info(f"{m:22s} n={len(sub)} acc={metrics[m]['_all']['acc']:.4f} wrong={metrics[m]['_all']['wrong']:.4f} "
                 f"({time.time() - t0:.0f}s)")
    save_json({"reader": a.reader, "dataset": a.dataset, "n_questions": len({it.get("qid", it["iid"]) for it in insts}),
               "n_instances": len(insts), "engine": llm.engine_info, "metrics": metrics,
               "dependence_cost": {k: dict(v) for k, v in dep_cost.items()},
               "claim_extraction_cost": claim_cost,
               "claim_extraction_accuracy": {k: float(np.mean(v)) for k, v in claim_acc.items()}}, out / "metrics.json")


if __name__ == "__main__":
    main()
